"""Write the submission and the resumable state archive."""

import json
import zipfile

from .checkpoint import STATE_FILE_NAME


def export_submission(submission, checkpoint, artifact_dir):
    def sort_id(item):
        identifier = str(item.get('id'))
        return (0, int(identifier)) if identifier.lstrip('-').isdigit() else (1, identifier)

    submission.sort(key=sort_id)

    errors = []
    for i, item in enumerate(submission):
        if not isinstance(item.get('id'), int) or isinstance(item.get('id'), bool):
            errors.append(f'item[{i}]: id must be an integer')
        docs = item.get('relevant_docs')
        if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
            errors.append(f'item[{i}]: relevant_docs must be list[str]')
            docs = []
        doc_set = set(docs)
        chunks = item.get('relevant_chunks')
        if not isinstance(chunks, list):
            errors.append(f'item[{i}]: relevant_chunks must be a list')
            continue
        for j, c in enumerate(chunks):
            if not isinstance(c, dict):
                errors.append(f'item[{i}].relevant_chunks[{j}]: not an object')
                continue
            if not isinstance(c.get('doc_id'), str):
                errors.append(f'item[{i}].relevant_chunks[{j}]: doc_id must be str')
            elif c['doc_id'] not in doc_set:
                errors.append(f'item[{i}].relevant_chunks[{j}]: doc_id not in relevant_docs')
            if not isinstance(c.get('chunk_text'), str) or not c['chunk_text'].strip():
                errors.append(f'item[{i}].relevant_chunks[{j}]: chunk_text empty')

    if errors:
        checkpoint.set_stage('submission', status='INVALID', errors=len(errors))
        raise ValueError(f'{len(errors)} submission format error(s): ' + ' | '.join(errors[:10]))

    sub_file = checkpoint.work_dir / 'submission.json'
    sub_file.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding='utf-8')
    zip_file = artifact_dir / 'submission.zip'
    temporary = zip_file.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_file, 'submission.json')
    temporary.replace(zip_file)
    print('FORMAT OK')
    print('submission:', sub_file, '(%.1f MB)' % (sub_file.stat().st_size / 1024**2))
    print('zip       :', zip_file)
    if submission:
        print('queries   :', len(submission))
        print('avg docs  : %.1f' % (sum(len(s['relevant_docs']) for s in submission) / len(submission)))
        print('avg chunks: %.1f' % (sum(len(s['relevant_chunks']) for s in submission) / len(submission)))

    checkpoint.set_stage('submission', status='DONE',
              queries=len(submission), file='submission.json',
              errors=len(errors), size_mb=round(sub_file.stat().st_size / 1024**2, 1))

    # show the manifest so the state is readable at a glance
    print()
    print('=' * 60)
    print((checkpoint.work_dir / STATE_FILE_NAME).read_text(encoding='utf-8'))

    return zip_file


def export_state(work_dir, artifact_dir, *, embedding_matrix=None):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bundle = artifact_dir / 'pa5_state.zip'
    temporary = bundle.with_suffix('.zip.partial')
    added = []

    if embedding_matrix is None:
        embedding_matrix = work_dir / 'doc_emb.npy'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name in (STATE_FILE_NAME, 'retrieval.jsonl', 'chunks.jsonl',
                     'retrieval_results.json', 'chunk_checkpoint.json',
                     'submission.json'):
            p = work_dir / name
            if p.exists():
                zf.write(p, 'pa5_state/' + name)
                added.append(name)

        if (work_dir / 'bm25').exists():
            for p in sorted((work_dir / 'bm25').rglob('*')):
                if p.is_file():
                    zf.write(p, 'pa5_state/' + p.relative_to(work_dir).as_posix())
            added.append('bm25/')

        marker = embedding_matrix.with_suffix('.done')
        if embedding_matrix.is_file() and marker.is_file():
            for p in (embedding_matrix, marker):
                zf.write(p, 'pa5_state/' + p.name)
                added.append(p.name)
        elif (work_dir / 'emb_shards').exists():
            for p in sorted((work_dir / 'emb_shards').glob('*.npy')):
                marker = p.with_suffix('.done')
                if marker.is_file():
                    for entry in (p, marker):
                        zf.write(entry, 'pa5_state/emb_shards/' + entry.name)
            added.append('emb_shards/')

    temporary.replace(bundle)
    print('bundle:', bundle, '(%.1f MB)' % (bundle.stat().st_size / 1024 ** 2))
    print('contains:', ', '.join(added))

    return bundle
