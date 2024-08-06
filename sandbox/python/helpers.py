import pandas as pd

pd.options.display.float_format = '{:.3f}'.format
def prittyprint(array):
    df = pd.DataFrame(array)
    print(df)
