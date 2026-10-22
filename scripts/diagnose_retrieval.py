#!/usr/bin/env python3
"""Inspect frozen predictions and routes without running a retriever or model."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tkh.corpus import load_graph,load_questions
from tkh.evaluation import entity_targets
from tkh.retrieval_diagnosis import diagnose_route,compare_rankings,verify_predictions
from tkh.divisive_hierarchy import scientific_edges


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def csv_rows(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                                   for k,v in row.items()})


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError(f'Diagnostic output already exists: {out}')
    tree=ROOT/'out/repair-tree/run1'
    predictions=verify_predictions(tree);protocol=read(tree/'protocol.json')
    lock=read(tree/'prediction_lock.json')
    if sha(tree/'protocol.json')!=lock['protocol_sha256']:raise ValueError('Tree protocol changed')
    frozen={};drift=[]
    for name,expected in protocol['frozen_sha256'].items():
        if sha(name)!=expected:raise ValueError(f'Tree frozen input changed: {name}')
        frozen[name]=expected
    graph=load_graph(ROOT/'data/tkh_collection10.json')
    questions=[q for q in load_questions(ROOT/'data/questions.csv') if q['type']=='A']
    nodes={n['id']:n for n in graph['nodes']};gold=read(ROOT/'data/ground_truth.json')
    targets={q['question_id']:entity_targets(gold[q['question_id']]['expected_methods'],graph) for q in questions}
    all_targets=sum(map(len,targets.values()));matched=sum(bool(t['node_ids']) for ts in targets.values() for t in ts)
    hierarchies={'bounded':ROOT/'out/repair-balanced-bounded-v2/snapshots/2026/hierarchy.json',
                 'novel':ROOT/'out/repair-balanced-novel-v2/snapshots/2026/hierarchy.json',
                 'legacy':ROOT/'results/snapshots/2026/hierarchy.json'}
    traces=read(tree/'traces.json');metrics=read(tree/'metrics.json');rows=[];by_question=[];summary={}
    for method,path in hierarchies.items():
        hierarchy=read(path);summary[method]={}
        for arm,rankings in predictions[method].items():
            channel=arm.split('::')[0];sums=Counter();leaves=[]
            for q in questions:
                qid=q['question_id'];baseline=predictions[method][channel+'::all'][qid]
                detail,counts=diagnose_route(targets[qid],baseline,rankings[qid],hierarchy,
                    traces[method]['queries'][arm][qid],nodes)
                sums.update(counts['stage_counts']);sums['gained']+=counts['gained'];sums['lost']+=counts['lost']
                leaves.append(counts['retained_records'])
                rows.extend({'method':method,'arm':arm,'question_id':qid,**r} for r in detail)
                by_question.append({'method':method,'arm':arm,'question_id':qid,**counts})
            expected=metrics['summary'][method][arm]
            if sums['hit']!=expected['hits'] or all_targets!=expected['targets']:
                raise ValueError('Route attribution disagrees with saved evaluation')
            summary[method][arm]={'hits':sums['hit'],'targets':all_targets,'matched_targets':matched,
                'stage_counts':{k:sums[k] for k in ('hit','unmatched','coarse_rejected','fine_rejected','retained_not_top_k')},
                'gained_over_flat':sums['gained'],'lost_from_flat':sums['lost'],
                'mean_retained_records':sum(leaves)/len(leaves)}
    overlay_rows=[];overlays={};identities={}
    for version,method in [('v2','bounded'),('v3','split')]:
        study=ROOT/f'out/repair-study-{version}';saved=verify_predictions(study);locks=read(study/'prediction_lock.json')
        provenance=locks['provenance']['frozen_sha256']
        for name,expected in provenance.items():
            if sha(name)==expected:frozen[name]=expected;continue
            snapshot=study/'source_snapshot/src/tkh'/Path(name).name
            if not snapshot.is_file() or sha(snapshot)!=expected:
                raise ValueError(f'Frozen input drift not explained by preserved source: {name}')
            drift.append({'study':version,'path':name,'historical_source_snapshot':str(snapshot),
                          'expected_sha256':expected,'current_sha256':sha(name)})
        hierarchy_path=ROOT/('out/repair-balanced-bounded-v2' if version=='v2' else 'out/repair-balanced-split-v3')/'snapshots/2026/hierarchy.json'
        if provenance.get(str(hierarchy_path))!=sha(hierarchy_path):raise ValueError('Assessed hierarchy changed')
        identities[version]={'method':method,'hierarchy_path':str(hierarchy_path),
            'hierarchy_sha256':sha(hierarchy_path),'predictions_sha256':sha(study/'predictions.json')}
        overlays[version]={}
        study_metrics=read(study/'metrics.json')
        for channel in ('bm25__native_sources','hybrid_no_tags'):
            counts=Counter();changed=0
            for q in questions:
                qid=q['question_id'];a=saved[method][channel+'::baseline'][qid];b=saved[method][channel+'::hierarchy'][qid]
                detail,pair=compare_rankings(targets[qid],a,b);counts.update(pair);changed+=a[:30]!=b[:30]
                overlay_rows.extend({'version':version,'channel':channel,'question_id':qid,**r} for r in detail)
            for arm,key in [('baseline','before_hits'),('hierarchy','after_hits')]:
                score=study_metrics['retrieval']['arms'][method][channel+'::'+arm]['tracks']['A']['metrics']['30']
                if counts[key]!=score['hits']:raise ValueError('Overlay comparison disagrees with saved evaluation')
            overlays[version][channel]={**counts,'questions_with_changed_top30':changed}
    bound=read(hierarchies['bounded']);type_counts=Counter(n['type'] for n in graph['nodes']);group_types=[]
    for g in bound['levels'][0]['groups']:
        counts=Counter(nodes[n]['type'] for n in g['members'])
        group_types.append({'id':g['id'],'members':len(g['members']),'types':dict(counts)})
    numeric={str(year):read(ROOT/f'out/repair-balanced-bounded-v2/snapshots/{year}/trace.json')['numeric_checks']
             for year in (2022,2026)}
    native,_=scientific_edges(graph,[n['id'] for n in graph['nodes'] if n['type'] not in {'article','author'}],gamma=.03)
    claims=Counter('positive' if e['weight']>0 else 'zero' for e in native if e['relation_type']=='claims')
    result={'schema_version':'saved-retrieval-diagnosis-v1','primary':'A micro Recall@30, original matching unchanged',
        'A_questions':len(questions),'targets':all_targets,'matched_target_occurrences':matched,
        'unmatched_target_occurrences':all_targets-matched,'method_identity':identities,
        'split_v3_same_final_hierarchy_as_novel_v2':sha(identities['v3']['hierarchy_path'])==sha(hierarchies['novel']),
        'overlays':overlays,'tree':summary,'questions':by_question,'coarse_type_counts':group_types,
        'corpus_type_counts':dict(type_counts),'objective_weights_from_trace':numeric,
        'claim_native_scientific_weights':dict(claims),
        'source_drift_preserved_historical_code':drift,
        'limitations':['Post-hoc development diagnosis, not a new retrieval experiment or final test.',
            'Unmatched means no exact surface or explicit alias; not proven absent from scientific content.',
            'Ranks absent from saved top100 lists are censored, not a measured full rank.',
            'Type concentration and weight imbalance are descriptive clues, not causal ablation evidence.',
            'Saved traces were not originally included in the prediction lock. Their path/membership consistency is checked now.',
            'No labels, aliases, weights, graph records or saved predictions were modified.']}
    sources=[tree/'predictions.json',tree/'traces.json',tree/'metrics.json',tree/'protocol.json',
             ROOT/'data/ground_truth.json',ROOT/'data/questions.csv',Path(__file__),
             ROOT/'src/tkh/retrieval_diagnosis.py',*hierarchies.values()]
    manifest={'files_sha256':{str(p):sha(p) for p in sources},
              'authenticated_frozen_inputs':len(frozen),'source_drift':drift}
    out.mkdir(parents=True)
    write(out/'metrics.json',result);write(out/'source_hashes.json',manifest)
    csv_rows(out/'targets.csv',rows);csv_rows(out/'overlay_targets.csv',overlay_rows)
    csv_rows(out/'unmatched.csv',[{'question_id':qid,**t} for qid,ts in targets.items() for t in ts if not t['node_ids']])
    lines=['# Где иерархия теряет ответы','',
        'Аудит сохранённых результатов. Поиск, модели и API не запускались; правила сопоставления не менялись.','',
        f'Проверены {len(questions)} вопросов A, {all_targets} целевых вхождений. Сопоставляются с узлами {matched}; '
        f'{all_targets-matched} не сопоставляются по точному имени/явному алиасу. Это не доказывает отсутствие информации в графе.','',
        '## Какая структура проверялась','',
        '`28/63 → 28/63` относится к `repair-balanced-bounded-v2` и также к более поздней `repair-balanced-split-v3`. '
        'Это добавление групповых переходов к поиску по всем записям. Начальное число оцениваемых узлов не сокращается. '
        'В BM25 все 14 списков Top-30 изменились, но ни одна целевая запись не добавилась и не исчезла.','',
        'В BM25+E5 добавление групповых переходов даёт 28→26: CHGNet в Q1 сдвигается с 30 на 32, '
        'M3GNet в Q12 — с 29 на 32. Новых попаданий нет.','',
        'Конечный JSON `split-v3` побайтно совпадает с `novel-v2`. Поэтому отдельный поиск сверху вниз по `novel-v2` '
        'оценивает то же конечное разбиение; это не новая независимая структура. '
        'Механизм обновления может отличаться, совпадение конечного среза не доказывает его эквивалентность во всех сценариях.','',
        '## Настоящий поиск сверху вниз','',
        'Сначала выбираются крупные группы и их дочерние группы; только затем оцениваются сохранённые узлы. '
        'Таблица ниже использует bounded-v2 и BM25+E5 с исходными гиперрёбрами и общими источниками.','',
        '| Крупных/мелких ветвей | В Top-30 | Нет сопоставления | Отсечены сверху | Отсечены ниже | Допущены, но вне Top-30 | Новые/потерянные попадания | Среднее число записей |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for arm in ('tree_2_8','tree_4_16','tree_8_32','all'):
        s=summary['bounded']['bm25_e5::'+arm];c=s['stage_counts']
        cells=[{'tree_2_8':'2/8','tree_4_16':'4/16','tree_8_32':'8/32','all':'все'}[arm],c['hit'],c['unmatched'],c['coarse_rejected'],c['fine_rejected'],c['retained_not_top_k'],
               f'{s["gained_over_flat"]}/{s["lost_from_flat"]}',f'{s["mean_retained_records"]:.1f}']
        lines.append('| '+' | '.join(map(str,cells))+' |')
    lines+=['','Сумма пяти колонок исходов в каждой строке равна 63. Новые/потерянные попадания считаются относительно '
        'полного поиска того же канала. Подсчёт сделан по вхождениям целей, а не числу разных названий.','',
        '## Потери при широком проходе 8/32','',
        '| Вопрос | Цель | Место в полном поиске | Место после обхода | Где потерялась | Крупные группы | Мелкие группы |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    lost=[r for r in rows if r['method']=='bounded' and r['arm']=='bm25_e5::tree_8_32' and r['change']=='lost']
    for r in lost:
        stage={'retained_not_top_k':'допущена, упала в ранжировании','fine_rejected':'отсечена мелкая группа','coarse_rejected':'отсечена крупная группа'}[r['stage']]
        cells=[r['question_id'],r['label'],r['before_rank'],r['after_rank'] or 'нет в сохранённом Top-100',stage,
               ', '.join(r['coarse_group_ids']),', '.join(r['fine_group_ids'])]
        lines.append('| '+' | '.join(map(str,cells))+' |')
    gained=[r for r in rows if r['method']=='bounded' and r['arm']=='bm25_e5::tree_8_32' and r['change']=='gained']
    lines+=['','Два новых попадания: '+ '; '.join(f'{r["question_id"]}: {r["label"]} '
        f'({r["before_rank"] or "вне сохранённого Top-100"} → {r["after_rank"]})' for r in gained)+'.','',
        '## Что именно подозрительно в структуре и поиске','',
        '1. В L0-bal000010 находятся 611 из 690 утверждений всего графа; 611 из 645 членов группы — утверждения. '
        'Это сильное разделение по форме записи. Оно не доказывает бессмысленности всех групп, но не похоже на '
        'тематическую папку, объединяющую методы с задачами и выводами.','',
        '2. Конструкция использует E5 от `type: surface_form`. В начальном срезе сумма весов семантических '
        'связей равна 1, исходных гиперрёбер — 0.03; в 2026 году — 3.1646 и 0.07091. Это суммы весов, '
        'не проценты реального влияния на разрез: вклад многоместного ребра зависит от числа затронутых групп. '
        'Причинное влияние требует отдельного сравнения.','',
        'Из 690 исходных рёбер типа `claims` 212 соединяют утверждение только со статьёй: после исключения '
        'метаданных у них остаётся один научный конец и нулевой вес разреза. Это не потеря исходного ребра '
        'в экспорте, но из него не получается научная связь с методом. Остальные 478 имеют положительный вес.','',
        '3. Выбор ветвей использует одну среднюю E5-точку и склеенный текст всех научных членов группы. '
        'Короткое имя метода может оказаться далеко от среднего, а задачи и длинные утверждения '
        'заполняют контекст. Новые API-подписи этот код вообще не читает.','',
        '4. После отсечения меняется и поиск внутри ветвей: нормировка BM25/E5 и исходные переходы '
        'ограничиваются выбранными узлами. Поэтому допуск правильного узла ещё не гарантирует попадания в Top-30.','',
        '5. Среди несопоставленных целей есть как общие сочетания подходов, так и явные проблемы записи имени. '
        'Например, эталон `JARVIS-DFT` не совпадает с узлом comp_00335 `JARVIS‑DFT` из-за разных дефисов. '
        'В графе есть meth_00221 `Universal HamGNN Hamiltonian model`, но нет точного имени/явного алиаса `HamGNN`. '
        'Это кандидаты на отдельный аудит сопоставления. Их нельзя молча исправить только в нашем результате: '
        'любое общее изменение проверяющего потребует повторного подсчёта всех сравниваемых методов.','',
        '## Следующее исправление','',
        'Сначала проверить один новый способ выбора ветвей при неизменном разбиении: несколько представителей '
        'научных подтем вместо одного среднего вектора, с тем же числом выбранных ветвей и тем же поиском листьев. '
        'Это отделит ошибку маршрутизации от ошибки самого разбиения. Представители выбираются только по '
        'членам группы и замороженным векторам, без вопросов, ответов и ручных тематических меток. '
        'До этого не запускать новые модели, генерацию подписей или пересборку всей иерархии.','',
        'Если замена представления ветви не решит потери, следующий предмет изменения — представление узла '
        'для группировки: собственное имя плюс научный контекст исходных гиперрёбер. Это отдельная гипотеза, '
        'пока не измеренный результат.','',
        '## Данные и ограничения','',
        f'Все численные результаты и проверки: `{out}/metrics.json`; полные вхождения целей: `targets.csv`; '
        'изменения при добавлении групповых связей: `overlay_targets.csv`; несопоставленные цели: `unmatched.csv`. '
        'Хеши входов: `source_hashes.json`.','',
        'Это разбор уже использованного набора для разработки. Никаких новых значений Recall или улучшений '
        'метода не заявлено. Пропуск ранга означает отсутствие в сохранённом Top-100, а не известный полный ранг. '
        'В v2 обновляющий код после эксперимента менялся; точные прежние версии сохранены в source_snapshot и '
        'проверены по исходным хешам. Хеши предсказаний, структур, данных и дерева поиска совпали с записями экспериментов.']
    report=out/'report.md';report.write_text('\n'.join(lines)+'\n')
    for path,expected in manifest['files_sha256'].items():
        if sha(path)!=expected:raise ValueError('Input changed during diagnosis')
    print(f'Saved diagnosis:{report}; {len(rows)} target/route records; no new retrieval or inference.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'out/retrieval-diagnosis-v1')
    args=p.parse_args();run(args.out)
