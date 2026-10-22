"""Evidence-bound tables and an English scientific article from saved metrics."""
from html import escape
import re
from pathlib import Path


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |',
        '| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def render_report(metrics,template):
    snapshots=metrics['construction']['snapshots'];transitions=metrics['construction']['transitions']
    arms=metrics['retrieval']['arms']
    def retrieval(base,condition):return arms[f'{base}::{condition}']['tracks']['A']['metrics']['30']
    def interval(x):return f'[{x[0]:.3f}, {x[1]:.3f}]'
    rows=[];previous=0
    for item in snapshots:
        s=item['snapshot'];counts=s['node_types']
        science=s['nodes']-counts.get('article',0)-counts.get('author',0)
        rows.append([s['cutoff'],s['nodes'],science,s['nodes']-previous,s['hyperedges'],
                     f"{s['arity']['median']:.0f} / {s['arity']['max']:g}"])
        previous=s['nodes']
    snap_table=table(['Year','Nodes','Science','Added','Edges','Arity med/max'],rows)
    final=metrics['coherence'][-1]['levels']
    coh_table=table(['Level','Cosine','Null mean','Excess'],
        [[name,f"{final[str(level)]['observed']:.4f}",f"{final[str(level)]['null_mean']:.4f}",
          f"{final[str(level)]['excess']:.4f}"] for level,name in enumerate(('Coarse','Fine'))])
    blind_table=table(['Regime','Found / judged','Rate','Wilson 95%'],
        [[('Coarse / distant' if s['level']==0 else 'Fine / distant' if s['regime']=='foreign_branch' else
           'Fine / sibling' if s['regime']=='sibling' else 'Random null'),
          f"{s['hits']}/{s['answered']}",f"{s['rate']:.1%}",interval(s['wilson95'])] for s in metrics['blind']['strata']])
    trans_table=table(['Transition','ARI coarse','ARI fine','Moves coarse/fine'],
        [[f"{t['from_cutoff']}–{t['to_cutoff']}",f"{t['levels'][0]['ari_shared_old_nodes']:.4f}",
          f"{t['levels'][1]['ari_shared_old_nodes']:.4f}",
          f"{t['levels'][0]['persistent_id_movement']} / {t['levels'][1]['persistent_id_movement']}"] for t in transitions])
    perturb=table(['Level','Mean ARI','Conditional t 95%'],
        [[('Coarse' if r['level']==0 else 'Fine'),f"{r['mean']:.4f}",interval(r['conditional_95_t_interval'])]
         for r in metrics['historical_perturbation']['levels']])
    calibration=table(['Native weight γ','Coarse excess','Fine excess'],
        [[f"{r['gamma']:g}",f"{r['quality']['0']['excess']:.4f}",
          f"{r['quality']['1']['excess']:.4f}"] for r in metrics['historical_calibration']['study']])
    names=[('bm25_context','BM25 / no walk'),('bm25__native','BM25 / native'),
           ('bm25__sources','BM25 / source'),('bm25__native_sources','BM25 / native + source'),
           ('hybrid_no_tags','BM25 + E5 / native + source'),('hybrid_graph','BM25 + E5 + tags / all layers')]
    ret_table=table(['Search / ordinary walk','Base','Groups','2/8','4/16'],
        [[label]+[f"{retrieval(base,c)['hits']}/{retrieval(base,c)['targets_all']}" for c in
            ('baseline','hierarchy','route_2_8','route_4_16')] for base,label in names])
    control_rows=[]
    for base,label in names[3:5]:
        values=[retrieval(base,f'null_{i}')['hits'] for i in range(5)]
        uncertainty=metrics['retrieval']['paired_question_uncertainty'][f'{base}::hierarchy']
        control_rows.append([label,f'{min(values)}–{max(values)}',retrieval(base,'identity_overlay')['hits'],
            f"{uncertainty['difference_micro_recall_at_30']:+.3f} {interval(uncertainty['paired_question_bootstrap_95_interval'])}"])
    ret_controls=table(['Seed / native + source','Random hits','Stay-put hits','ΔRecall [paired 95%]'],
        [[('BM25' if i==0 else 'BM25 + E5')]+row[1:] for i,row in enumerate(control_rows)])
    req_table=table(['Task','Evidence / remaining gap'],[
        ['T1','Snapshot counts, arity, types and quality issues retained'],
        ['T2 / T4','Native criterion, counted quotient, laminarity and budgets implemented'],
        ['T3','Checkpoint updates measured; real-temporal uncertainty / adaptive edits incomplete'],
        ['T5','Unmet: explanatory labels and independently measured over-claim rate'],
        ['T6','Independent nulls, perturbations and retrieval measured; useful hierarchy not shown'],
        ['T7 / gate','English report; actual build → validate → metrics → generated tables']])
    final_nodes=snapshots[-1]['snapshot']['nodes'];protocol=metrics['construction']['protocol']
    editable_fractions=[]
    for item in snapshots[1:]:
        s=item['snapshot'];counts=s['node_types']
        science=s['nodes']-counts.get('article',0)-counts.get('author',0)
        editable_fractions.append(len(item['growth_and_cost_trace']['editable_ids'])/science)
    replacements={'DATE':'20 October 2026','SNAPSHOTS':snap_table,'COHERENCE':coh_table,'BLIND':blind_table,
        'TRANSITIONS':trans_table,'PERTURBATION':perturb,'CALIBRATION':calibration,'RETRIEVAL':ret_table,'RETRIEVAL_CONTROLS':ret_controls,
        'REQUIREMENTS':req_table,'BROAD_BUDGET':protocol['budgets'][0],'FINE_BUDGET':protocol['budgets'][1],
        'GAMMA':protocol['gamma'],'KNN':15,'A_QUESTIONS':arms['bm25_context::baseline']['tracks']['A']['questions'],
        'A_TARGETS':retrieval('bm25_context','baseline')['targets_all'],'ANSWER_NODES':metrics['answer_nodes'],
        'QUALITY_ERRORS':metrics['data_quality_errors'],'TOTAL_NODES':final_nodes,
        'BM25_GRAPH_HITS':retrieval('bm25__native_sources','baseline')['hits'],
        'BM25_HIER_HITS':retrieval('bm25__native_sources','hierarchy')['hits'],
        'DENSE_GRAPH_HITS':retrieval('hybrid_no_tags','baseline')['hits'],
        'DENSE_HIER_HITS':retrieval('hybrid_no_tags','hierarchy')['hits'],
        'INTEGRITY_VIEWS':len(snapshots),'UPDATE_COUNT':len(transitions),
        'EDITABLE_MIN':f'{min(editable_fractions):.2%}','EDITABLE_MAX':f'{max(editable_fractions):.2%}'}
    rendered=template
    for key,value in replacements.items():rendered=rendered.replace('{{'+key+'}}',str(value))
    if re.search(r'\{\{[^}]+\}\}',rendered):raise ValueError('Unresolved report evidence placeholder')
    return rendered


def figures(metrics,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arms=metrics['retrieval']['arms'];conditions=('baseline','hierarchy','route_2_8','route_4_16')
    labels=('Baseline','Group walk','Route 2/8','Route 4/16')
    colors=('#555555','#0072B2','#D55E00','#009E73')
    with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
        from matplotlib.patches import FancyBboxPatch
        fig,ax=plt.subplots(figsize=(3.7,3.2),layout='constrained')
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        def box(x,y,w,h,label,color='#f2f2f2'):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',
                facecolor=color,edgecolor='#333333',linewidth=.8))
            ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
        def arrow(start,end):
            ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':.9,'color':'#333333'})
        box(.02,.84,.44,.13,'Own record text\nFrozen E5 + kNN')
        box(.54,.84,.44,.13,'Original hyperedges\nSource-balanced weights')
        box(.15,.61,.70,.13,'Native cut: semantic cost + hyperedge cost')
        arrow((.24,.84),(.35,.75));arrow((.76,.84),(.65,.75))
        budgets=metrics['construction']['protocol']['budgets']
        box(.04,.32,.55,.17,f'At most {budgets[0]} coarse groups\n↓ one parent per fine group\nAt most {budgets[1]} fine groups total','#e9f1f7')
        arrow((.36,.61),(.32,.50))
        box(.65,.32,.32,.17,'Counted quotient\nOne multiway edge\n+ endpoint counts')
        arrow((.59,.40),(.64,.40))
        box(.04,.06,.55,.13,'Original record leaves\nNative relations remain addressable')
        arrow((.32,.32),(.32,.20))
        box(.65,.04,.32,.19,'Next year\nLoad checkpoint\nAppend evidence\nRefine edit region','#f5ede5')
        arrow((.81,.32),(.81,.24))
        for ext in ('png','pdf'):
            fig.savefig(directory/f'structure.{ext}',dpi=220,metadata={'Title':'Conceptual method schematic; no measured values except configured budgets'})
        plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(3.7,3.3),layout='constrained')
        levels=metrics['coherence'][-1]['levels'];x=np.arange(2);w=.32
        for offset,key,label,color in ((-w/2,'observed','Real groups','#0072B2'),
                                      (w/2,'null_mean','Matched null','#999999')):
            bars=axes[0].bar(x+offset,[levels[str(i)][key] for i in range(2)],w,
                label=label,color=color,edgecolor='black',linewidth=.3,
                hatch='' if key=='observed' else '//')
            axes[0].bar_label(bars,fmt='%.3f',fontsize=7,padding=1)
        axes[0].set(xticks=x,xticklabels=['Coarse','Fine'],ylim=(0,.26),ylabel='Mean cosine')
        axes[0].set_title('(a) Independent-family group coherence',loc='left',fontsize=9)
        axes[0].legend(fontsize=7,loc='upper left',ncol=2,frameon=False)
        transitions=metrics['construction']['transitions'];x=np.arange(len(transitions))
        for i,label,color,marker in ((0,'Coarse','#0072B2','o'),(1,'Fine','#D55E00','s')):
            axes[1].plot(x,[t['levels'][i]['ari_shared_old_nodes'] for t in transitions],
                color=color,marker=marker,label=label,linewidth=1.2,markersize=4)
        axes[1].set(xticks=x,xticklabels=[f"{str(t['from_cutoff'])[2:]}→{str(t['to_cutoff'])[2:]}" for t in transitions],
            ylim=(0,1.08),ylabel='Shared-old-node ARI')
        axes[1].set_title('(b) Real annual transitions',loc='left',fontsize=9)
        axes[1].legend(fontsize=7,loc='lower right',ncol=2,frameon=False)
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        for ext in ('png','pdf'):
            fig.savefig(directory/f'quality.{ext}',dpi=220,metadata={'Title':'Independent coherence and chronological stability; metrics.json source'})
        plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(3.7,3.1),layout='constrained')
        for ax,base,title in zip(axes,('bm25__native_sources','hybrid_no_tags'),
                ('BM25 + native/source','BM25 + E5 + native/source'),strict=True):
            values=[100*arms[f'{base}::{condition}']['tracks']['A']['metrics']['30']['recall_all_micro'] for condition in conditions]
            bars=ax.bar(np.arange(4),values,color=colors,edgecolor='black',linewidth=.3)
            ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],fontsize=8,padding=1)
            ax.set(ylim=(0,56),ylabel='Recall@30 (%)',xticks=np.arange(4),
                   xticklabels=['Base','Groups','2/8','4/16'])
            ax.set_title(title,loc='left',fontsize=9)
            ax.spines[['top','right']].set_visible(False)
        for ext in ('png','pdf'):fig.savefig(directory/f'retrieval.{ext}',dpi=200,metadata={'Title':'Matched hierarchy retrieval; all 63 A targets'})
        plt.close(fig)


def write_pdf(markdown,path,root):
    """Render an article with a full-width title/abstract and two-column body.

    Equations are drawn as vector glyphs from Matplotlib's mathtext layout;
    neither a TeX installation nor bitmap formula screenshots are required.
    """
    import hashlib
    import matplotlib
    from matplotlib import mathtext,font_manager
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY,TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,
        Spacer,Table,TableStyle,Image,Flowable,FrameBreak,CondPageBreak,KeepTogether,
        BalancedColumns,NextPageTemplate,PageBreak)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);root=Path(root)
    margin=43;gutter=18;page_width,page_height=A4
    full_width=page_width-2*margin;column_width=(full_width-gutter)/2
    body=ParagraphStyle('body',fontName='Times-Roman',fontSize=9.8,leading=12.0,
        alignment=TA_JUSTIFY,spaceAfter=6,splitLongWords=True,allowWidows=0,allowOrphans=0)
    styles={'body':body,
        'title':ParagraphStyle('title',fontName='Times-Bold',fontSize=17.5,leading=20.5,
            alignment=TA_CENTER,spaceAfter=7),
        'subtitle':ParagraphStyle('subtitle',fontName='Times-Roman',fontSize=9,leading=11,
            alignment=TA_CENTER,spaceAfter=10),
        'abstract_head':ParagraphStyle('abstract_head',fontName='Times-Bold',fontSize=10.5,
            leading=12,alignment=TA_CENTER,spaceAfter=4),
        'abstract':ParagraphStyle('abstract',parent=body,fontSize=9.4,leading=11.4,spaceAfter=6),
        'h2':ParagraphStyle('h2',fontName='Times-Bold',fontSize=11.5,leading=14,
            spaceBefore=8,spaceAfter=5,keepWithNext=True),
        'h3':ParagraphStyle('h3',fontName='Times-Bold',fontSize=10,leading=12,
            spaceBefore=5,spaceAfter=4,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=7.6,leading=9),
        'cellhead':ParagraphStyle('cellhead',fontName='Times-Bold',fontSize=7.3,leading=8.6),
        'caption':ParagraphStyle('caption',fontName='Times-Roman',fontSize=8.5,leading=10.2,spaceAfter=4),
        'reference':ParagraphStyle('reference',fontName='Times-Roman',fontSize=8.5,leading=10.3,
            spaceAfter=5,splitLongWords=True)}

    def markup(text):
        text=escape(text).replace('→','&rarr;').replace('²','<super>2</super>')
        text=text.replace('Δ','<font name="Symbol">D</font>')
        text=text.replace('γ','<font name="Symbol">g</font>')
        text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
        text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
        text=re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#163b63">\1</link>',text)
        text=re.sub(r'(?<!["=])(https?://[^\s<>]+)',
            lambda m:f'<link href="{m.group(1)}" color="#163b63">{m.group(1)}</link>',text)
        return text

    class Equation(Flowable):
        def __init__(self,expressions,number):
            super().__init__();self.number=number
            with matplotlib.rc_context({'mathtext.fontset':'stix'}):
                parser=mathtext.MathTextParser('path')
                self.parts=[parser.parse('$'+expression+'$',dpi=72,
                    prop=font_manager.FontProperties(size=10.5)) for expression in expressions]
            self.scale=min(1.,(column_width-20)/max(p.width for p in self.parts))
            self.height=sum(float(p.height)*self.scale+5 for p in self.parts)+8
            self.width=column_width
        def draw(self):
            canvas=self.canv;y=self.height-4
            for part in self.parts:
                y-=float(part.height)*self.scale
                origin_x=(self.width-18-float(part.width)*self.scale)/2
                baseline=y+float(part.depth)*self.scale
                for font,size,code,x,y_offset in part.glyphs:
                    name='math-'+hashlib.sha256(font.fname.encode()).hexdigest()[:12]
                    if name not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont(name,font.fname))
                    canvas.setFont(name,size*self.scale)
                    canvas.drawString(origin_x+x*self.scale,baseline+y_offset*self.scale,chr(code))
                for x,y_rect,w,h in part.rects:
                    canvas.rect(origin_x+x*self.scale,baseline+y_rect*self.scale,w*self.scale,h*self.scale,stroke=0,fill=1)
                y-=5
            canvas.setFont('Times-Roman',9)
            canvas.drawRightString(self.width,self.height/2-2,f'({self.number})')

    if '<!-- ARTICLE BODY -->' not in markdown:raise ValueError('Article template must separate title/abstract from body')
    front,main=markdown.split('<!-- ARTICLE BODY -->',1)
    front_blocks=[b.strip() for b in front.split('\n\n') if b.strip()]
    header=[]
    for block in front_blocks:
        if block.startswith('# '):header.append(Paragraph(markup(block[2:]),styles['title']))
        elif block.startswith('## '):header.append(Paragraph(markup(block[3:]),styles['abstract_head']))
        elif block.startswith('Research assessment'):header.append(Paragraph(markup(block),styles['subtitle']))
        else:header.append(Paragraph(markup(' '.join(block.splitlines())),styles['abstract']))
    header_height=sum(f.wrap(full_width,10000)[1]+f.getSpaceBefore()+f.getSpaceAfter() for f in header)+9
    top=page_height-40;bottom=39;first_top=top-header_height-12
    def frame(x,y,w,h,name):return Frame(x,y,w,h,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    first_frames=[frame(margin,top-header_height,full_width,header_height,'title-abstract'),
        frame(margin,bottom,column_width,first_top-bottom,'first-left'),
        frame(margin+column_width+gutter,bottom,column_width,first_top-bottom,'first-right')]
    regular_frames=[frame(margin,bottom,column_width,top-bottom,'left'),
        frame(margin+column_width+gutter,bottom,column_width,top-bottom,'right')]
    final_frames=[frame(margin,bottom,full_width,top-bottom,'balanced-final')]
    def footer(canvas,doc):
        canvas.setFont('Times-Roman',8)
        canvas.drawCentredString(page_width/2,21,str(doc.page))
    doc=BaseDocTemplate(str(path),pagesize=A4,leftMargin=margin,rightMargin=margin,
        topMargin=40,bottomMargin=bottom,invariant=1,
        title=front_blocks[0].lstrip('# '),author='',
        subject='Research assessment: native temporal hypergraph abstraction; measured findings and limitations')
    doc.addPageTemplates([PageTemplate(id='first',frames=first_frames,onPage=footer,autoNextPageTemplate='body'),
                         PageTemplate(id='body',frames=regular_frames,onPage=footer),
                         PageTemplate(id='last',frames=final_frames,onPage=footer)])
    story=header+[FrameBreak()];prefix=None;lines=main.splitlines();i=0;equation_number=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- ARTICLE END -->':
            prefix=story;story=[];i+=1;continue
        if line=='$$':
            expressions=[];i+=1
            while i<len(lines) and lines[i].strip()!='$$':
                if lines[i].strip():expressions.append(lines[i].strip())
                i+=1
            if i==len(lines):raise ValueError('Unclosed display equation')
            i+=1;equation_number+=1;story.append(Equation(expressions,equation_number));continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[cell.strip() for cell in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',cell) for cell in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);weights=[1.]*n
            if n==3:weights=[.85,1,1.7]
            elif n==4:weights=[1.45,1,1,1.55]
            elif n==5:weights=[2.5,1,1,1,1]
            elif n==6:weights=[.85,1,1,1,1,1.25]
            if 'ΔRecall' in rows[0][-1]:weights=[1.15,.8,.8,1.9]
            widths=[column_width*w/sum(weights) for w in weights]
            cells=[[Paragraph(markup(c),styles['cellhead'] if row_index==0 else styles['cell'])
                    for c in row] for row_index,row in enumerate(rows)]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.7,colors.black),
                ('LINEBELOW',(0,0),(-1,0),.4,colors.black),
                ('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2.5),
                ('RIGHTPADDING',(0,0),(-1,-1),2.5),('TOPPADDING',(0,0),(-1,-1),3),
                ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='caption':
                caption=story.pop();story.append(KeepTogether([caption,t,Spacer(1,8)]))
            else:story.extend([t,Spacer(1,8)])
            continue
        match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if match:
            from PIL import Image as PILImage
            image_path=root/match.group(2)
            with PILImage.open(image_path) as raster:ratio=raster.height/raster.width
            picture=Image(str(image_path),width=column_width,height=column_width*ratio)
            i+=1
            while i<len(lines) and not lines[i].strip():i+=1
            if i<len(lines) and lines[i].startswith('**Figure '):
                caption=Paragraph(markup(lines[i].strip()),styles['caption']);i+=1
                story.append(KeepTogether([picture,Spacer(1,4),caption,Spacer(1,6)]))
            else:story.append(picture)
            continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'))
            story.append(CondPageBreak(55))
            story.append(Paragraph(markup(line.lstrip('#').strip()),styles['h2' if level==2 else 'h3']))
            i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('|','#','<!--','![','$$')):
            paragraph.append(lines[i].strip());i+=1
        text=' '.join(paragraph)
        style=(styles['caption'] if text.startswith('**Table ') else
               styles['reference'] if re.match(r'^\[\d+\]',text) else styles['body'])
        story.append(Paragraph(markup(text),style))
    if prefix is not None:
        story=prefix+[NextPageTemplate('last'),PageBreak(),
            BalancedColumns(story,nCols=2,needed=1,innerPadding=gutter/2,leftPadding=0,
                            rightPadding=0,topPadding=0,bottomPadding=0)]
    doc.build(story)
