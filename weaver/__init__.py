"""

"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


#--- Weave (Functional Software Interface)

from lupa import LuaRuntime, unpacks_lua_table, as_attrgetter


SANDBOX = r'''
function(untrusted_code, callbacks)
    local env = {};
    for key, value in python.iterex(callbacks.items()) do
        env[key] = value
    end
    env.print = print

    local untrusted_function, message = load(untrusted_code, nil, 't', env)
    if not untrusted_function then return nil, message end
    debug.sethook(function() error("timeout") end, "", 1e6)
    local success, ret = pcall(untrusted_function)
    debug.sethook()
    return success, ret
end
''' #/SANDBOX


@dataclass
class Weaver:
    registry: Dict[str, Callable]

    @classmethod
    def create(cls):
        registry = {}
        
        return cls(
            registry=registry,
        )

    def register(self, func=None, /, *, name=None, unpack=False):
        def wrapper(func, /, name=name):
            if name is None:
                name = func.__name__
            
            if unpack:
                func = unpacks_lua_table(func)

            assert name not in self.registry, \
                f'Registry already contains name: {name!r}'

            self.registry[name] = func

            return func
        
        if func is not None:
            return wrapper(func)

        return wrapper
    
    def execute(self, code):
        lua = LuaRuntime(
            unpack_returned_tuples=True,
        )

        sandbox = lua.eval(SANDBOX)

        success, ret = sandbox(
            code,
            as_attrgetter(self.registry),
        )

        if not success:
            if not isinstance(ret, Exception):
                ret = Exception(f'{ret!r}')
            raise ret

        ret = self.realize(ret)
        return ret

    @staticmethod
    def realize(x):
        def _realize(x):
            try:
                keys = list(x.keys())
            except:
                return x
            else:
                if 1 in keys:
                    return [_realize(x[i]) for i in range(1, 1+len(x))]
                else:
                    return { k: _realize(v) for k, v in x.items() }
        
        return _realize(x)


weaver = Weaver.create()


#--- Braid (Parallel Flow Visualization)

from typing import NewType, Union, Tuple, Dict, TypedDict
from datetime import datetime

from .braid import Integrator, Seconds, Pressure, Latitude, Longitude

SeedTuple = NewType('SeedTuple', Tuple[Pressure, Latitude, Longitude])
SeedDict = TypedDict('SeedDict', prs=Pressure, lat=Latitude, lng=Longitude)

#     *args: Union[List[SeedTuple], List[SeedDict]],
#     **kwargs: TypedDict(None, {
#         'from': Seconds,
#         'to': Seconds,
#         'seeds': Union[List[SeedTuple], List[SeedDict]],
#     }, total=False),
# ):


_g_braid_integrator: Integrator = None


@weaver.register(name='braid', unpack=True)
def _weave_braid_integrate(**kwargs):
    seeds = kwargs.pop('seeds')
    from_ = kwargs.pop('from')
    to = kwargs.pop('to')

    match from_:
        case str():
            from_ = datetime.strptime(from_, "%Y-%m-%d")
            from_ = from_ - datetime.strptime('2012-01-01', '%Y-%m-%d')
            from_ = from_.total_seconds()
    
    match to:
        case str():
            to = datetime.strptime(to, "%Y-%m-%d")
            to = to - datetime.strptime('2012-01-01', '%Y-%m-%d')
            to = to.total_seconds()
    
    seeds = Weaver.realize(seeds)

    print(f'{seeds = !r} {from_ = !r} {to = !r}')

    traces = []
    for seed in seeds:
        match seed:
            case {'prs': prs, 'lng': lng, 'lat': lat}:
                seed = (prs, lng, lat)

        print(f'{from_ = !r} {seed = !r} {to = !r}')

        trace = []
        for t, y in _g_braid_integrator.integrate(    
            t0=from_,
            y0=seed,
            tf=to,
        ):
            sec = t
            prs, lng, lat = y

            trace.append({
                'sec': sec,
                'prs': prs,
                'lng': lng,
                'lat': lat,
            })

        traces.append(trace)
    
    return traces


#--- Spool (Distributed Object Store)

from uuid import uuid4 as uuid


_g_spools: Dict[str, Spool] = {}


def _make_token(prefix):
    return f'{prefix}-{uuid()!s}'


@dataclass
class Spool:
    name: str
    format: name
    ro_token: str = field(repr=False)
    rw_token: str = field(repr=False)

    @classmethod
    def create(cls, name, format):
        ro_token = _make_token('ro')
        rw_token = _make_token('rw')

        return cls(
            name=name,
            format=format,
            ro_token=ro_token,
            rw_token=rw_token,
        )
    
    def emit(self, *values):
        print(f'struct.pack({self.format!r}, *{values!r})')


@weaver.register(name='create', unpack=True)
def _weaver_spool_create(name, format):
    spool = Spool.create(
        name=name,
        format=format,
    )

    _g_spools[spool.ro_token] = spool
    _g_spools[spool.rw_token] = spool

    return {
        'tokens': {
            'ro': spool.ro_token,
            'rw': spool.rw_token,
        },
    }


@weaver.register(name='emit', unpack=True)
def _weaver_spool_emit(spool, *values):
    match spool:
        case str() as token:
            spool = _g_spools[token]
        
        case {'tokens': {'rw': token}}:
            spool = _g_spools[token]
        
        case {'tokens': {'ro': token}}:
            spool = _g_spools[token]

    values = Weaver.realize(values)

    spool.emit(*values)


#--- Weave (HTTP Server Interface)

from base64 import urlsafe_b64decode as b64decode

from flask import Flask, request


app = Flask(__name__)


@app.route('/weave/', methods=['POST'])
def weave():
    code = request.args.get('code')
    code = b64decode(code)
    code = code.decode('ascii')

    return weaver.execute(code)


def main(bind, host, port, debug):
    global _g_braid_integrator

    _g_braid_integrator = Integrator.from_files(
        ugrd=Path.cwd() / 'data' / 'UGRD-144x73.dat',
        vgrd=Path.cwd() / 'data' / 'VGRD-144x73.dat',
        vvel=Path.cwd() / 'data' / 'VVEL-144x73.dat',
    )

    app.run(
        host=bind,
        port=port,
        debug=debug,
    )


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=7772)
    parser.add_argument('--debug', action='store_true')
    args = vars(parser.parse_args())

    main(**args)


if __name__ == '__main__':
    cli()
