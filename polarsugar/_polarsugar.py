from typing import Callable
from functools import reduce
import polars as pl

class ForkAccessor:
    def __init__(self, expr: pl.Expr) -> None:
        '''
        A collection of methods to split a `pl.Expr` into multiple
        and potentially summarize them
        '''
        self._expr = expr

    def __call__(self, funclist:'list[str|Callable[[pl.Expr], pl.Expr]]', name_sep='_') -> 'list[pl.Expr]':
        '''
        Transform the original `pl.Expr` into multiple with a list of functions.
        `funclist` can contain both real callables and string names,
        the latter is interpreted as `pl.Expr.<method name>`
        '''
        res = []
        for f in funclist:
            if isinstance(f, str):
                try:
                    res.append(
                        getattr(self._expr, f)().alias(f'{self._expr.meta.output_name()}{name_sep}{f}')
                    )
                except AttributeError:
                    raise ValueError(f'string "{f}" does not represent a `pl.Expr` method name') from None
            else:
                res.append(f(self._expr))
        return res

    def all(self, funclist:'list[str|Callable[[pl.Expr], pl.Expr]]', name_sep='_') -> pl.Expr:
        '''
        Do the `fork()` and boolean-`&` the results
        '''
        return reduce(lambda acc, el: acc & el, self.__call__(funclist, name_sep))

    def any(self, funclist:'list[str|Callable[[pl.Expr], pl.Expr]]', name_sep='_') -> pl.Expr:
        '''
        Do the `fork()` and boolean-`|` the results
        '''
        return reduce(lambda acc, el: acc | el, self.__call__(funclist, name_sep))



class colsugar:
    def __init__(self, expr: 'pl.Expr|str'):
        '''
        a wrapper around `pl.Expr` that provides missing syntactic sugar
        '''
        if isinstance(expr, str):
            expr = pl.col(expr)
        self._expr = expr

    @property
    def fork(self):
        '''
        A collection of methods to split a `pl.Expr` into multiple
        and potentially summarize them
        '''
        return ForkAccessor(self._expr)

    def pipe(self, func: 'Callable[[pl.Expr],pl.Expr]', *args, **kwargs) -> pl.Expr:
        '''
        apply a `func` to `self` to transform the expression
        '''
        res = func(self._expr, *args, **kwargs)
        if not isinstance(res, pl.Expr):
            raise TypeError('`func` returned something other than `pl.Expr`')
        return res

    def dictmap(self, dictionary:dict) -> pl.Expr:
        '''
        map values in the corresponding `pl.Series` from `dictionary`s keys to values
        a.k.a. `pd.Series.map`
        '''
        mapper = pl.DataFrame({
            'keys': list(dictionary.keys()),
            'values': list(dictionary.values())
        })
        return self._expr.map(
            lambda s: s.to_frame('keys').join(mapper, on='keys',how='left').to_series(1)
        )


def register(name='sugar'):
    '''
    register a set of accessors for the utility classes in this package
    '''
    pl.api.register_expr_namespace(name)(colsugar)
