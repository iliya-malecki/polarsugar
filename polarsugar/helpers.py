'''For stuff that i dont know how to organize in a namespace'''

import polars as pl

def pyselector(df: pl.DataFrame, dtype: pl.DataType):
    '''
    dunno if pl.selectors does selection of arbitrary dtypes. i needed it for pl.Struct
    '''
    return [col for col, dt in zip(df.columns, df.dtypes) if isinstance(dt, dtype)]

def unwrap(x: pl.DataFrame|pl.Series):
    '''
    recursively unnest struct columns that presumably come from a
    json column parsed into a struct with `.str.json_extract()`
    '''

    if isinstance(x, pl.Series):
        return unwrap(x.struct.unnest())

    structs = pyselector(x, pl.Struct)
    if not structs:
        return x

    return unwrap(
        pl.concat(
            [
                x.drop(structs),
                *(x[struct].struct.unnest().select(pl.all().prefix(struct+'.')) for struct in structs)
            ],
            how='horizontal'
        )
    )
