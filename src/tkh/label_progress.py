"""Terminal display kept separate from generation and evidence handling."""


def label_progress(requests,cached_groups,total_groups):
    from tqdm import tqdm
    return tqdm(total=requests,desc='Naming',unit='request',dynamic_ncols=True,
        miniters=1,smoothing=0,postfix={'groups':f'{cached_groups}/{total_groups}'},
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} requests '
                   '[{elapsed} elapsed, ETA {remaining}]{postfix}')
