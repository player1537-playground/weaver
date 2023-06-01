"""

"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


#--- Weave (Functional Software Interface)

import threading

from lupa import LuaRuntime, unpacks_lua_table, as_attrgetter


SANDBOX = r'''
function(untrusted_code, callbacks)
    local env = {};
    for key, value in python.iterex(callbacks.items()) do
        env[key] = value
    end
    env.assert=assert; env.error=error; env.ipairs=ipairs; env.next=next;
    env.pairs=pairs; env.pcall=pcall; env.print=print; env.select=select;
    env.tonumber=tonumber; env.tostring=tostring; env.type=type;
    env.unpack=unpack; env._VERSION=_VERSION; env.xpcall=xpcall;
    env.string = {
        byte=string.byte, char=string.char, find=string.find,
        format=string.format, gmatch=string.gmatch, gsub=string.gsub,
        len=string.len, lower=string.lower, match=string.match, rep=string.rep,
        reverse=string.reverse, sub=string.sub, upper=string.upper,
    }
    env.table = {
        insert=table.insert, maxn=table.maxn, remove=table.remove,
        sort=table.sort,
    }
    env.math = {
        abs=math.abs, acos=math.acos, asin=math.asin, atan=math.atan,
        atan2=math.atan2, ceil=math.ceil, cos=math.cos, cosh=math.cosh,
        deg=math.deg, exp=math.exp, floor=math.floor, fmod=math.fmod,
        frexp=math.frexp, huge=math.huge, ldexp=math.ldexp, log=math.log,
        log10=math.log10, max=math.max, min=math.min, modf=math.modf,
        pi=math.pi, pow=math.pow, rad=math.rad, sin=math.sin, sinh=math.sinh,
        sqrt=math.sqrt, tan=math.tan, tanh=math.tanh,
    }

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
    context: threading.local

    @classmethod
    def create(cls):
        registry = {}
        context = threading.local()
        
        return cls(
            registry=registry,
            context=context,
        )

    def register(self, func=None, /, *, name=None, unpack=False, repack=True):
        def wrapper(func, /, name=name):
            if name is None:
                name = func.__name__
            
            if unpack:
                func = unpacks_lua_table(func)
            
            if repack:
                def func(*args, func=func, **kwargs):
                    ret = func(*args, **kwargs)
                    ret = self.encode(ret)
                    return ret

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
        self.context.lua = lua

        registry = { **self.registry }
        registry = as_attrgetter(registry)

        sandbox = lua.eval(SANDBOX)

        success, ret = sandbox(
            code,
            registry,
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
    
    def encode(self, x):
        def _encode(x):
            if isinstance(x, list) or isinstance(x, tuple):
                x = [_encode(v) for v in x]
                x = self.context.lua.table(*x)
            elif isinstance(x, dict):
                x = { k: _encode(v) for k, v in x.items() }
                x = self.context.lua.table(**x)
            return x
        
        return _encode(x)


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


@weaver.register(name='braid', unpack=True, repack=True)
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

    # print(f'{seeds = !r} {from_ = !r} {to = !r}')

    traces = []
    for seed in seeds:
        match seed:
            case {'prs': prs, 'lng': lng, 'lat': lat}:
                seed = (prs, lng, lat)

        # print(f'{from_ = !r} {seed = !r} {to = !r}')

        points = []
        for t, y in _g_braid_integrator.integrate(    
            t0=from_,
            y0=seed,
            tf=to,
        ):
            sec = t
            prs, lng, lat = y

            points.append({
                'sec': sec,
                'prs': prs,
                'lng': lng,
                'lat': lat,
            })

        traces.append({
            'seed': seed,
            'points': points,
        })

    return traces


#--- Spool (Distributed Object Store)

from uuid import uuid4 as uuid
from threading import Lock
import struct

_g_spools: Dict[str, Spool] = {}


def _make_token(prefix):
    return f'{prefix}-{uuid()!s}'


@dataclass
class Spool:
    name: str
    columns: List[str]
    format: name
    ro_token: str = field(repr=False)
    rw_token: str = field(repr=False)
    data: List[bytes] = field(repr=False)
    lock: Lock = field(repr=False)

    @classmethod
    def create(cls, name, **kwargs):
        columns = []
        formats = []
        for column, format in kwargs.items():
            columns.append(column)
            formats.append(format)
        
        format = ''.join(formats)

        ro_token = _make_token('ro')
        rw_token = _make_token('rw')
        data = []
        lock = Lock()

        return cls(
            name=name,
            columns=columns,
            format=format,
            ro_token=ro_token,
            rw_token=rw_token,
            data=data,
            lock=lock,
        )
    
    def emit(self, **values):
        values = [values[column] for column in self.columns]
        data = struct.pack(self.format, *values)
        with self.lock:
            self.data.append(data)

    def items(self):
        with self.lock:
            data = b''.join(self.data)
            self.data = [data]
        
        size = struct.calcsize(self.format)
        items = []
        for i in range(0, len(data), size):
            item = data[i:i+size]
            item = struct.unpack(self.format, item)
            item = { column: value for column, value in zip(self.columns, item) }
            items.append(item)

        return items


@weaver.register(name='create', unpack=True)
def _weaver_spool_create(name, **kwargs):
    spool = Spool.create(
        name=name,
        **kwargs,
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
def _weaver_spool_emit(spool, **values):
    match spool:
        case str() as token:
            spool = _g_spools[token]
        
        case {'tokens': {'rw': token}}:
            spool = _g_spools[token]
        
        case {'tokens': {'ro': token}}:
            raise ValueError(f'Cannot emit to a spool with a read-only token')

    # print(f'{values=!r}')
    values = Weaver.realize(values)
    # print(f'{values=!r}')

    spool.emit(**values)


@weaver.register(name='items', unpack=True, repack=True)
def _weave_spool_items(spool):
    match spool:
        case str() as token:
            spool = _g_spools[token]
        
        case {'tokens': {'rw': token}}:
            spool = _g_spools[token]
        
        case {'tokens': {'ro': token}}:
            spool = _g_spools[token]

    return spool.items()



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
