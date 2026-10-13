#!/usr/bin/env python3
"""Collect closed Polymarket Sports markets and hourly trade summaries.

This uses public read-only endpoints; it does not need an API key. It is built
for a long-running research collection: discovery is paginated into a compact
JSONL catalogue and timestamped trades are immediately converted to hourly
aggregates. Raw trade responses are discarded by default. Re-running the same
command resumes rather than re-fetching completed market summaries.

Important limitation: the Data API returns at most `--trade-limit` records for
a market.  A response at that limit is flagged as potentially truncated; it is
not evidence that a market had exactly that many trades.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'data' / 'all_closed_sports'
GAMMA = 'https://gamma-api.polymarket.com'
DATA = 'https://data-api.polymarket.com'
SPORTS_TAG_ID = '1'
USER_AGENT = 'school-polymarket-sports-collector/1.0'


def get_json(url: str, retries: int = 4):
    """Fetch JSON with bounded exponential backoff for public API rate limits."""
    error = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={'User-Agent': USER_AGENT})
            with urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception as exc:  # HTTP errors also belong in the resumable log.
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f'{type(error).__name__}: {error}')


def jsonl_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                try:
                    ids.add(str(json.loads(line)['id']))
                except (KeyError, json.JSONDecodeError):
                    continue
    return ids


MARKET_FIELDS = (
    'id', 'conditionId', 'slug', 'question', 'startDate', 'endDate', 'closedTime',
    'outcomes', 'outcomePrices', 'clobTokenIds', 'volume', 'volumeNum',
    'volumeClob', 'volumeAmm', 'liquidity', 'liquidityNum', 'liquidityClob',
    'liquidityAmm', 'spread', 'bestBid', 'bestAsk', 'lastTradePrice',
)


def compact_market(market: dict) -> dict:
    """Keep research variables, not large descriptions/media/event payloads."""
    return {field: market.get(field) for field in MARKET_FIELDS if field in market}


def read_discovery_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def write_discovery_checkpoint(path: Path, checkpoint: dict) -> None:
    """Atomically persist the cursor only after its page is safely catalogued."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(checkpoint, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


def read_trade_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def write_trade_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(checkpoint, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


def pending_retry_markets(retry_events_path: Path) -> dict[str, dict]:
    """Replay the retry journal to recover markets still needing a fetch."""
    pending: dict[str, dict] = {}
    if not retry_events_path.exists():
        return pending
    with retry_events_path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            market_id = str(event.get('id', ''))
            if not market_id:
                continue
            if event.get('event') == 'error' and isinstance(event.get('market_ref'), dict):
                pending[market_id] = event['market_ref']
            elif event.get('event') == 'resolved':
                pending.pop(market_id, None)
    return pending


def legacy_log_frontier(log_path: Path, catalogue: Path) -> int:
    """Locate the furthest catalogue row touched by the pre-checkpoint run."""
    if not log_path.exists():
        return 0
    highest_id = -1
    with log_path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                market_id = int(json.loads(line).get('id', -1))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            highest_id = max(highest_id, market_id)
    if highest_id < 0:
        return 0
    row_index = 0
    with catalogue.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                row_index += 1
                continue
            try:
                market_id = int(json.loads(line).get('id', -1))
            except (json.JSONDecodeError, TypeError, ValueError):
                return row_index
            row_index += 1
            if market_id >= highest_id:
                return row_index
    return row_index


def bootstrap_trade_resume(out: Path, total_markets: int) -> tuple[dict, int]:
    """Migrate a legacy run by indexing missing outputs and its log frontier."""
    catalogue = out / 'markets.jsonl'
    hourly_dir = out / 'hourly_activity_24h'
    checkpoint_path = out / 'trade_checkpoint.json'
    retry_events_path = out / 'trade_retry_events.jsonl'
    checkpoint = read_trade_checkpoint(checkpoint_path)
    if checkpoint.get('resume_version') == 2:
        return checkpoint, min(int(checkpoint.get('next_catalogue_index', 0)), total_markets)

    frontier = min(legacy_log_frontier(out / 'trade_collection_log.jsonl', catalogue), total_markets)
    retry_events_path.parent.mkdir(parents=True, exist_ok=True)
    queued = 0
    # Old logs did not retain market metadata on failures. Inspect the already
    # attempted prefix once and queue only markets whose summary file is absent.
    with catalogue.open(encoding='utf-8') as source, retry_events_path.open('a', encoding='utf-8') as retry_log:
        for row_index, line in enumerate(source):
            if row_index >= frontier:
                break
            if not line.strip():
                continue
            try:
                market = json.loads(line)
            except json.JSONDecodeError:
                continue
            market_id = str(market.get('id', ''))
            if market_id and market.get('conditionId') and not (hourly_dir / f'{market_id}.json.gz').exists():
                retry_log.write(json.dumps({'event': 'error', 'id': market_id,
                                            'market_ref': market,
                                            'detail': 'queued during legacy resume migration'},
                                           separators=(',', ':')) + '\n')
                queued += 1
        retry_log.flush()
    checkpoint = {'resume_version': 2, 'next_catalogue_index': frontier,
                  'catalogue_size': total_markets, 'updated_at_unix': time.time()}
    write_trade_checkpoint(checkpoint_path, checkpoint)
    print(f'legacy run indexed: resume at catalogue row {frontier:,}/{total_markets:,}; '
          f'queued {queued:,} unsaved earlier markets for retry', flush=True)
    return checkpoint, frontier


def discover(out: Path, page_size: int, max_pages: int | None, dry_run: bool):
    """Keyset-page through the public Sports catalogue and append unique rows."""
    catalogue = out / 'markets.jsonl'
    checkpoint_path = out / 'discovery_checkpoint.json'
    known = jsonl_ids(catalogue)
    checkpoint = read_discovery_checkpoint(checkpoint_path)
    page = int(checkpoint.get('pages_completed', 0))
    cursor = checkpoint.get('next_cursor')
    seen_cursors = set()
    if dry_run:
        print(f'DRY RUN: would page closed markets with tag_id={SPORTS_TAG_ID} into {catalogue}')
        return
    if cursor:
        print(f'resuming discovery at keyset page {page + 1}', flush=True)
    while max_pages is None or page < max_pages:
        params = {'closed': 'true', 'tag_id': SPORTS_TAG_ID, 'limit': page_size,
                  'order': 'id', 'ascending': 'true'}
        if cursor:
            params['after_cursor'] = cursor
        response = get_json(f'{GAMMA}/markets/keyset?{urlencode(params)}')
        if not isinstance(response, dict):
            raise RuntimeError('Unexpected keyset response; expected an object with markets and next_cursor.')
        markets = response.get('markets', [])
        if not isinstance(markets, list):
            raise RuntimeError('Unexpected keyset response; markets was not a list.')
        if not markets:
            break
        added = 0
        with catalogue.open('a', encoding='utf-8') as handle:
            for market in markets:
                market_id = str(market.get('id', ''))
                if not market_id or market_id in known:
                    continue
                handle.write(json.dumps(compact_market(market), separators=(',', ':')) + '\n')
                known.add(market_id)
                added += 1
        page += 1
        print(f'discovery keyset_page={page} received={len(markets)} new={added} total={len(known)}', flush=True)
        next_cursor = response.get('next_cursor')
        write_discovery_checkpoint(checkpoint_path, {
            'pages_completed': page,
            'next_cursor': next_cursor,
            'markets_catalogued': len(known),
            'updated_at_unix': time.time(),
        })
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise RuntimeError('Keyset cursor repeated; refusing to loop over the same page.')
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        time.sleep(.15)
    print(f'discovery complete: {len(known)} unique closed Sports markets in {catalogue}', flush=True)


def read_catalogue(path: Path) -> list[dict]:
    rows, seen = [], set()
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run --stage discover first.')
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            market = json.loads(line)
            market_id = str(market.get('id', ''))
            if market_id and market_id not in seen:
                rows.append(market); seen.add(market_id)
    return rows


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)
    except ValueError:
        return None


def hourly_activity(market: dict, trades: list[dict], hours: int = 24) -> list[dict]:
    """Aggregate timestamped trades into the final `hours` pre-close bins.

    The returned trade endpoint contains executions, not historical order-book
    snapshots. Thus price and activity are valid retrospective hourly measures;
    liquidity and bid/ask spread are intentionally not estimated here.
    """
    close_time = parse_utc(market.get('endDate'))
    if close_time is None:
        return []
    close_unix = close_time.timestamp()
    # A binary market has separate YES and NO token trades.  Keep them apart:
    # combining their prices would make an hourly probability meaningless.
    buckets: dict[tuple[int, str, str], list[dict]] = {}
    for trade in trades:
        try:
            seconds_before = close_unix - float(trade['timestamp'])
            hour_before = int((seconds_before - 1e-9) // 3600) + 1
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= hour_before <= hours:
            key = (hour_before, str(trade.get('asset', '')), str(trade.get('outcome', '')))
            buckets.setdefault(key, []).append(trade)

    rows = []
    for (hour_before, asset, outcome), values in sorted(buckets.items()):
        sizes, notionals, prices = [], [], []
        wallets, transactions = set(), set()
        buy_records = sell_records = 0
        for trade in values:
            try:
                size, price = float(trade['size']), float(trade['price'])
            except (KeyError, TypeError, ValueError):
                continue
            sizes.append(size); prices.append(price); notionals.append(size * price)
            if trade.get('proxyWallet'):
                wallets.add(trade['proxyWallet'])
            if trade.get('transactionHash'):
                transactions.add(trade['transactionHash'])
            buy_records += trade.get('side') == 'BUY'
            sell_records += trade.get('side') == 'SELL'
        if not sizes:
            continue
        prices.sort()
        middle = len(prices) // 2
        median_price = prices[middle] if len(prices) % 2 else (prices[middle - 1] + prices[middle]) / 2
        rows.append({
            'market_id': str(market['id']),
            'condition_id': market.get('conditionId'),
            'slug': market.get('slug'),
            'question': market.get('question'),
            'end_time_utc': close_time.isoformat(),
            'hours_before_close': hour_before,
            'hour_start_utc': datetime.fromtimestamp(close_unix - hour_before * 3600, tz=timezone.utc).isoformat(),
            'hour_end_utc': datetime.fromtimestamp(close_unix - (hour_before - 1) * 3600, tz=timezone.utc).isoformat(),
            'asset': asset,
            'outcome': outcome,
            'outcome_index': values[0].get('outcomeIndex'),
            'trade_records': len(values),
            'unique_transaction_hashes': len(transactions),
            'unique_wallets_observed': len(wallets),
            'buy_records': buy_records,
            'sell_records': sell_records,
            'share_size': sum(sizes),
            # The endpoint can return both sides of an execution. This is a
            # trade-side total, not an exchange-verified market volume.
            'trade_side_notional_usdc': sum(notionals),
            'vwap_price': sum(notionals) / sum(sizes),
            'median_price': median_price,
        })
    return rows


def fetch_trade_file(market: dict, trades_dir: Path | None, hourly_dir: Path,
                     trade_limit: int, keep_raw: bool, dry_run: bool) -> dict:
    market_id = str(market['id'])
    activity_path = hourly_dir / f'{market_id}.json.gz'
    if activity_path.exists():
        try:
            with gzip.open(activity_path, 'rt', encoding='utf-8') as handle:
                json.load(handle)
            return {'id': market_id, 'status': 'cached'}
        except (OSError, EOFError, json.JSONDecodeError):
            # A hard kill during an older non-atomic write may leave a partial
            # file. Keep it for diagnosis, then replace it after a good fetch.
            corrupt_path = activity_path.with_suffix(activity_path.suffix + '.corrupt')
            if corrupt_path.exists():
                corrupt_path = activity_path.with_name(f'{market_id}.{int(time.time())}.json.gz.corrupt')
            activity_path.rename(corrupt_path)
    path = trades_dir / f'{market_id}.json.gz' if trades_dir else None
    if path and path.exists():
        # Raw files collected before hourly aggregation was added can be
        # upgraded locally without another API request.
        if not activity_path.exists() and not dry_run:
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as handle:
                    payload = json.load(handle)
                with gzip.open(activity_path, 'wt', encoding='utf-8') as handle:
                    json.dump(hourly_activity(market, payload.get('trades', [])), handle, separators=(',', ':'))
                return {'id': market_id, 'status': 'cached', 'hourly_rebuilt': True}
            except Exception as exc:
                return {'id': market_id, 'status': 'error', 'detail': f'cached-file rebuild: {type(exc).__name__}: {exc}'}
        return {'id': market_id, 'status': 'cached'}
    condition_id = market.get('conditionId')
    if not condition_id:
        return {'id': market_id, 'status': 'skipped', 'detail': 'missing conditionId'}
    if dry_run:
        return {'id': market_id, 'status': 'would fetch'}
    try:
        query = urlencode({'market': condition_id, 'limit': trade_limit, 'takerOnly': 'false'})
        trades = get_json(f'{DATA}/trades?{query}')
        if not isinstance(trades, list):
            raise ValueError('trade response was not a JSON list')
        temporary_activity = activity_path.with_suffix(activity_path.suffix + '.tmp')
        with gzip.open(temporary_activity, 'wt', encoding='utf-8') as handle:
            json.dump(hourly_activity(market, trades), handle, separators=(',', ':'))
        temporary_activity.replace(activity_path)
        if keep_raw:
            payload = {'market_id': market_id, 'condition_id': condition_id,
                       'retrieved_at_unix': time.time(), 'trade_limit': trade_limit,
                       'history_possibly_truncated': len(trades) >= trade_limit, 'trades': trades}
            assert path is not None
            with gzip.open(path, 'wt', encoding='utf-8') as handle:
                json.dump(payload, handle, separators=(',', ':'))
        return {'id': market_id, 'status': 'downloaded', 'trades': len(trades),
                'possibly_truncated': len(trades) >= trade_limit}
    except Exception as exc:
        return {'id': market_id, 'status': 'error', 'detail': f'{type(exc).__name__}: {exc}'}


def collect_trades(out: Path, workers: int, trade_limit: int, max_markets: int | None,
                   keep_raw: bool, dry_run: bool):
    catalogue = out / 'markets.jsonl'
    if not catalogue.exists():
        raise FileNotFoundError(f'Missing {catalogue}; run --stage discover first.')
    with catalogue.open(encoding='utf-8') as handle:
        total_markets = sum(1 for line in handle if line.strip())
    if max_markets is not None:
        total_markets = min(total_markets, max_markets)
    trades_dir = out / 'trades' if keep_raw else None
    if trades_dir:
        trades_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir = out / 'hourly_activity_24h'
    hourly_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / 'trade_collection_log.jsonl'
    checkpoint_path = out / 'trade_checkpoint.json'
    retry_events_path = out / 'trade_retry_events.jsonl'
    checkpoint, start_index = bootstrap_trade_resume(out, total_markets)
    if max_markets is not None and start_index:
        raise ValueError('Cannot combine --max-markets with an existing trade checkpoint.')
    print(f'resuming at catalogue row {start_index + 1:,}/{total_markets:,} '
          f'(next row checkpointed: {start_index:,})', flush=True)

    summary = {'cached': 0, 'downloaded': 0, 'error': 0, 'skipped': 0,
               'would fetch': 0, 'possibly_truncated': 0}
    max_in_flight = max(workers * 4, workers)
    processed = 0
    next_index = start_index
    catalogue_index = 0  # Nonblank market rows; matches total_markets/checkpoint.
    pending_info: dict = {}
    completed_indexes: set[int] = set()
    retry_markets = pending_retry_markets(retry_events_path)

    def record_result(future, log, retry_log):
        nonlocal processed
        record = future.result()
        market, index, _was_retry = pending_info.pop(future)
        market_id = str(record.get('id', market.get('id', '')))
        if record['status'] == 'error':
            retry_markets[market_id] = market
            retry_log.write(json.dumps({'event': 'error', 'id': market_id,
                                        'market_ref': market,
                                        'detail': record.get('detail')},
                                       separators=(',', ':')) + '\n')
            retry_log.flush()
        elif market_id in retry_markets:
            retry_markets.pop(market_id, None)
            retry_log.write(json.dumps({'event': 'resolved', 'id': market_id},
                                       separators=(',', ':')) + '\n')
            retry_log.flush()
        log.write(json.dumps(record, separators=(',', ':')) + '\n')
        log.flush()
        status = record['status']
        summary[status] = summary.get(status, 0) + 1
        summary['possibly_truncated'] += int(bool(record.get('possibly_truncated')))
        processed += 1
        if index is not None:
            completed_indexes.add(index)
        if status == 'error' and (summary['error'] <= 5 or summary['error'] % 100 == 0):
            print(f"trade error market={market_id} detail={record.get('detail')}", flush=True)

    def save_checkpoint():
        write_trade_checkpoint(checkpoint_path, {
            'resume_version': 2,
            'next_catalogue_index': next_index,
            'catalogue_size': total_markets,
            'updated_at_unix': time.time(),
        })

    def advance_frontier() -> None:
        nonlocal next_index
        while next_index in completed_indexes:
            completed_indexes.remove(next_index)
            next_index += 1

    def report_progress(previous_frontier: int) -> None:
        if next_index // 1000 > previous_frontier // 1000:
            print(f'trades progress={next_index:,}/{total_markets:,} '
                  f'({next_index / max(total_markets, 1):.1%}) {summary}', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool, \
            log_path.open('a', encoding='utf-8') as log, \
            retry_events_path.open('a', encoding='utf-8') as retry_log:
        futures = set()
        with catalogue.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                current_index = catalogue_index
                catalogue_index += 1
                if current_index >= total_markets:
                    break
                if current_index < start_index:
                    continue
                market = json.loads(line)
                future = pool.submit(fetch_trade_file, market, trades_dir, hourly_dir,
                                     trade_limit, keep_raw, dry_run)
                futures.add(future)
                pending_info[future] = (market, current_index, False)
                if len(futures) >= max_in_flight:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    old_frontier = next_index
                    for future in done:
                        record_result(future, log, retry_log)
                    advance_frontier()
                    save_checkpoint()
                    report_progress(old_frontier)
            for future in futures:
                record_result(future, log, retry_log)
            old_frontier = next_index
            advance_frontier()
            save_checkpoint()
            report_progress(old_frontier)

        # Retry previous failures after forward collection, once per run. A
        # retry failure remains journaled for a later invocation.
        if next_index >= total_markets and retry_markets:
            retry_list = list(retry_markets.values())
            print(f'retrying {len(retry_list):,} previously failed markets once', flush=True)
            retry_futures = set()
            for market in retry_list:
                future = pool.submit(fetch_trade_file, market, trades_dir, hourly_dir,
                                     trade_limit, keep_raw, dry_run)
                retry_futures.add(future)
                pending_info[future] = (market, None, True)
                if len(retry_futures) >= max_in_flight:
                    done, retry_futures = wait(retry_futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        record_result(future, log, retry_log)
            for future in retry_futures:
                record_result(future, log, retry_log)
    print(f'trade pass complete: catalogue row {next_index:,}/{total_markets:,} '
          f'({next_index / total_markets:.1%}); processed this run={processed:,} {summary}; '
          f'pending retries={len(retry_markets):,}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['discover', 'trades', 'all'], default='all')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--page-size', type=int, default=100)
    parser.add_argument('--max-pages', type=int, default=None, help='Safety cap for discovery testing.')
    parser.add_argument('--max-markets', type=int, default=None, help='Safety cap for trade testing.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trade-limit', type=int, default=10_000)
    parser.add_argument('--keep-raw', action='store_true',
                        help='Also archive each raw trade response (uses much more disk).')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.page_size <= 100:
        parser.error('--page-size must be 1–100 (Gamma endpoint maximum)')
    if args.workers < 1 or args.workers > 12:
        parser.error('--workers must be 1–12')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ('discover', 'all'):
        discover(args.out, args.page_size, args.max_pages, args.dry_run)
    if args.stage in ('trades', 'all'):
        if args.dry_run and not (args.out / 'markets.jsonl').exists():
            print('DRY RUN: trade stage needs an existing catalogue; dry-run discovery does not write one.')
            return
        collect_trades(args.out, args.workers, args.trade_limit, args.max_markets,
                       args.keep_raw, args.dry_run)


if __name__ == '__main__':
    main()
