r"""?????????

???????? / demucs / wav2vec2 emission / Whisper ???????????
????? tools/_bench_cache/????? finalize() ???????
???????????????????

?????????:
    # 1. ????????? 7 ???
    .\.venv-whisperx\Scripts\python.exe tools\align_bench.py cache --all
    .\.venv-whisperx\Scripts\python.exe tools\align_bench.py cache "??"

    # 2. ??????? songs.json ??????????
    .\.venv-whisperx\Scripts\python.exe tools\align_bench.py eval --all

?????????????????????:
    .\.venv-whisperx\Scripts\python.exe tools\align_bench.py clean
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CACHE = os.path.join(HERE, '_bench_cache')
SONGS = os.path.join(ROOT, 'songs.json')


def load_songs():
    with open(SONGS, encoding='utf-8') as f:
        data = json.load(f)
    return data, (data['songs'] if isinstance(data, dict) else data)


def safe_name(title):
    return ''.join(c if c.isalnum() else '_' for c in title)[:40]


def prep_texts(song):
    import add_song_new as asn
    raw_all = [l.get('src') or asn.strip_ruby(l['text']) for l in song['lines']]
    join_prev, raw_texts = [], []
    for r in raw_all:
        j, body = asn.parse_join_prev(r)
        join_prev.append(j)
        raw_texts.append(body)
    plains = [asn.display_text(r) for r in raw_texts]
    align_texts = [asn.alignment_text(r) for r in raw_texts]
    return plains, align_texts, join_prev


# ------------------------------------------------------------------ ???

def build_cache(song, device='cpu'):
    import add_song_new as asn
    import aligner

    title = song['title']
    path = os.path.join(CACHE, safe_name(title) + '.npz')
    meta_path = os.path.join(CACHE, safe_name(title) + '.json')
    if os.path.exists(path) and os.path.exists(meta_path):
        print(f"  [cache] ??????: {title}")
        return

    video_id = song.get('videoId')
    if not video_id:
        print(f"  [cache] ?? videoId???: {title}")
        return

    print(f"\n=== ???: {title} ===", flush=True)
    audio_path = asn.download_audio(video_id)
    audio = asn.load_wav(audio_path)
    audio_len = len(audio) / aligner.SAMPLE_RATE

    plains, align_texts, join_prev = prep_texts(song)
    lang = aligner.detect_language(plains, None)
    name = aligner.HF_ALIGN_MODELS.get(lang)

    vocals, separated = aligner.separate_vocals(audio, device=device)

    tokenizer, feat, model = aligner.load_ctc_model(name, device=device)
    vocab = tokenizer.get_vocab()
    blank_id = tokenizer.pad_token_id
    if blank_id is None:
        blank_id = vocab.get('<pad>', 0)
    delim = getattr(tokenizer, 'word_delimiter_token', '|')
    word_delim_id = vocab.get(delim)

    targets, token_line, per_line_kept, dropped = aligner.build_targets(
        align_texts, lang, vocab, blank_id, word_delim_id)
    used = sorted(set(targets) | {blank_id})
    col_of = {t: i for i, t in enumerate(used)}
    emission = aligner.compute_emissions(vocals, model, feat, device=device,
                                         keep_tokens=used)

    # Whisper ???????? words?????????????
    import stable_whisper
    ct = 'int8' if device == 'cpu' else 'float16'
    wmodel = stable_whisper.load_faster_whisper('large-v3', device=device,
                                                compute_type=ct)
    try:
        res = wmodel.transcribe_stable(vocals, language=lang, vad=True)
    except Exception:
        res = wmodel.transcribe_stable(vocals, language=lang, vad=False)
    words = [{'text': w.word, 'start': w.start, 'end': w.end}
             for seg in res.segments for w in seg.words]

    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(
        path,
        emission=emission.astype(np.float32),
        vocals=vocals.astype(np.float32),
        tgt_cols=np.array([col_of[t] for t in targets], dtype=np.int32),
        token_line=np.array(token_line, dtype=np.int32),
    )
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'title': title, 'lang': lang, 'audio_len': audio_len,
            'blank_col': col_of[blank_id], 'separated': bool(separated),
            'words': words, 'plains': plains, 'align_texts': align_texts,
            'join_prev': join_prev,
        }, f, ensure_ascii=False)
    print(f"  [cache] OK: {title}", flush=True)


def load_cache(title):
    path = os.path.join(CACHE, safe_name(title) + '.npz')
    meta_path = os.path.join(CACHE, safe_name(title) + '.json')
    if not (os.path.exists(path) and os.path.exists(meta_path)):
        return None
    z = np.load(path)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    return z, meta


# ------------------------------------------------------- ????? finalize

def run_from_cache(title, verbose=False):
    import aligner
    import pykakasi

    got = load_cache(title)
    if got is None:
        return None
    z, meta = got
    plains = meta['plains']
    n_lines = len(plains)
    lang = meta['lang']
    kks = pykakasi.kakasi() if lang == 'ja' else None

    emission = z['emission']
    token_line = z['token_line'].tolist()
    tgt_cols = z['tgt_cols'].tolist()
    total_frames = emission.shape[0]

    char_time, lyr_line = aligner.anchors_from_words(
        meta['words'], meta['align_texts'], lang, kks, verbose=verbose)
    anchor_bounds = aligner.anchor_line_bounds(char_time, lyr_line, n_lines,
                                               verbose=verbose)
    expected = aligner.expected_frames_from_anchors(
        anchor_bounds, token_line, n_lines, total_frames)

    import re
    char_counts = [len(re.sub(r'\s+', '', p)) for p in plains]
    results = aligner.finalize(
        emission, z['vocals'], tgt_cols, token_line, n_lines,
        meta['blank_col'], meta['audio_len'], anchor_bounds=anchor_bounds,
        expected_frames=expected, separated=meta['separated'],
        char_counts=char_counts, join_prev=meta['join_prev'],
        verbose=verbose)
    return results, meta


# ------------------------------------------------------------------ ??

def evaluate(titles):
    """? songs.json ????????????????????"""
    _, songs = load_songs()
    by_title = {s['title']: s for s in songs}

    print(f"\n{'??':<26} {'??':>4} {'???':>7} {'???':>7} "
          f"{'>2s':>4} {'??':>4} {'interp':>6}")
    print("-" * 74)
    total_big = 0
    for t in titles:
        got = run_from_cache(t)
        if got is None:
            print(f"{t:<26}  (???)")
            continue
        results, meta = got
        old = [(l.get('time'), l.get('end')) for l in by_title[t]['lines']]
        diffs = []
        for i, r in enumerate(results):
            if i < len(old) and old[i][0] is not None and r['time'] is not None:
                diffs.append(abs(r['time'] - old[i][0]))
        med = float(np.median(diffs)) if diffs else 0.0
        mx = max(diffs) if diffs else 0.0
        big = sum(1 for d in diffs if d > 2.0)
        total_big += big
        short = sum(1 for r in results if r['time'] is not None
                    and (r['end'] - r['time']) < 0.5)
        interp = sum(1 for r in results if (r.get('source') or 'interp') == 'interp')
        print(f"{t[:24]:<26} {len(results):>4} {med:>7.2f} {mx:>7.2f} "
              f"{big:>4} {short:>4} {interp:>6}")
    print("-" * 74)
    print(f"  ??????? >2s ????: {total_big}")


def cached_titles():
    if not os.path.isdir(CACHE):
        return []
    out = []
    for fn in os.listdir(CACHE):
        if fn.endswith('.json'):
            with open(os.path.join(CACHE, fn), encoding='utf-8') as f:
                out.append(json.load(f)['title'])
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith('--')]
    all_flag = '--all' in sys.argv

    if cmd == 'clean':
        if os.path.isdir(CACHE):
            shutil.rmtree(CACHE)
            print(f"??? {CACHE}")
        else:
            print("??????")
        return

    _, songs = load_songs()
    if cmd == 'cache':
        targets = songs if all_flag else [s for s in songs
                                          if s['title'] in args]
        for s in targets:
            build_cache(s)
    elif cmd == 'eval':
        titles = cached_titles() if all_flag else args
        evaluate(titles)
    else:
        raise SystemExit(f"????: {cmd}")


if __name__ == '__main__':
    main()
