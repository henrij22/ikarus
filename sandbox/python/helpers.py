import pandas as pd

def prettyprint(array, prec=3):
    # pd.options.display.float_format = f'{{:.{prec}f}}'
    pd.options.display.float_format = f'{{:.{prec}f}}'.format
    df = pd.DataFrame(array)
    print(df)
