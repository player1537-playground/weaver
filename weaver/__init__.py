"""

"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


#--- Utilities (for all of the following codes)

def _make_token(prefix):
    return f'{prefix}-{uuid()!s}'


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


@weaver.register(name='weave_braid_trace', unpack=True, repack=True)
def _weave_braid_integrate(**kwargs):
    seeds = kwargs.pop('seeds')
    from_ = kwargs.pop('from')
    to = kwargs.pop('to')

    if isinstance(from_, str):
        from_ = datetime.strptime(from_, "%Y-%m-%d")
        from_ = from_ - datetime.strptime('2012-01-01', '%Y-%m-%d')
        from_ = from_.total_seconds()
    
    if isinstance(to, str):
        to = datetime.strptime(to, "%Y-%m-%d")
        to = to - datetime.strptime('2012-01-01', '%Y-%m-%d')
        to = to.total_seconds()
    
    seeds = Weaver.realize(seeds)

    # print(f'{seeds = !r} {from_ = !r} {to = !r}')

    traces = []
    for seed in seeds:
        if isinstance(seed, dict) and 'prs' in seed and 'lng' in seed and 'lat' in seed:
            seed = (seed['prs'], seed['lng'], seed['lat'])

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


def lookup_spool(spool) -> Tuple[Spool, bool]:
    spool = Weaver.realize(spool)

    if isinstance(spool, str):
        token = spool
    
    elif isinstance(spool, dict) and 'tokens' in spool and isinstance(spool['tokens'], dict) and 'rw' in spool['tokens'] and isinstance(spool['tokens']['rw'], str):
        token = spool['tokens']['rw']

    elif isinstance(spool, dict) and 'tokens' in spool and isinstance(spool['tokens'], dict) and 'ro' in spool['tokens'] and isinstance(spool['tokens']['ro'], str):
        token = spool['tokens']['ro']
    
    elif isinstance(spool, dict) and 'rw' in spool and isinstance(spool['rw'], str):
        token = spool['rw']

    elif isinstance(spool, dict) and 'ro' in spool and isinstance(spool['ro'], str):
        token = spool['ro']
    
    else:
        raise KeyError(f'No spool found for: {spool=!r}')

    spool = _g_spools[token]
    return spool, token == spool.rw_token


def _make_spool_token(prefix):
    return _make_token(f'{prefix}-spool')


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
    def create(cls, name, formats: List[Tuple[str, str]]):
        the_formats = formats

        columns, formats = [], []
        for column, format in the_formats:
            columns.append(column)
            formats.append(format)
        
        format = ''.join(formats)

        ro_token = _make_spool_token('ro')
        rw_token = _make_spool_token('rw')
        data = []
        lock = Lock()

        print(dict(
            name=name,
            columns=columns,
            format=format,
            ro_token=ro_token,
            rw_token=rw_token,
            data=data,
            lock=lock,
        ))

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

    def dump(self, path: Path):
        with self.lock:
            data = b''.join(self.data)
            self.data = [data]
        
        path.write_bytes(data)


@weaver.register(name='weave_spool_create', unpack=True)
def _weaver_spool_create(name, *args, **kwargs):
    formats = []
    for arg in args:
        arg = Weaver.realize(arg)

        if isinstance(arg, dict):
            for k, v in arg.items():
                formats.append((k, v))
        else:
            raise ValueError(f'Unexpected argument: create({name!r}, *{args!r}, **{kwargs!r}')
    
    for k, v in kwargs.items():
        formats.append((k, v))

    spool = Spool.create(
        name=name,
        formats=formats,
    )

    _g_spools[spool.ro_token] = spool
    _g_spools[spool.rw_token] = spool

    return {
        'tokens': {
            'ro': spool.ro_token,
            'rw': spool.rw_token,
        },
    }


@weaver.register(name='weave_spool_emit', unpack=True)
def _weaver_spool_emit(spool, **values):
    spool, can_write = lookup_spool(spool)
    if not can_write:
        raise ValueError(f'Cannot emit to a spool with a read-only token')

    # print(f'{values=!r}')
    values = Weaver.realize(values)
    # print(f'{values=!r}')

    spool.emit(**values)


@weaver.register(name='weave_spool_items', unpack=True, repack=True)
def _weave_spool_items(spool):
    spool, _can_write = lookup_spool(spool)
    return spool.items()


#--- Graph (Large Graph Visualization)

_g_graphs: Dict[str, Graph] = {}


def _make_graph_token(prefix):
    return _make_token(f'{prefix}-graph')


@dataclass
class Attribute:
    graph: Graph
    name: str
    path: Path


@dataclass
class Graph:
    root: Path
    fs_token: str
    ro_token: str
    rw_token: str
    lock: Lock
    attributes: Dict[str, Path]

    @classmethod
    def create(cls, root: Path):
        fs_token = _make_graph_token('fs')
        ro_token = _make_graph_token('ro')
        rw_token = _make_graph_token('rw')

        root = root / fs_token
        root.mkdir(exist_ok=False, parents=False)

        lock = Lock()
        attributes = {}

        return cls(
            root=root,
            fs_token=fs_token,
            ro_token=ro_token,
            rw_token=rw_token,
            lock=lock,
            attributes=attributes,
        )
    
    def ingest(self, index, attributes):
        with self.lock:
            for name, spool in [
                ('index', index),
            ] + [
                (k, v)
                for k, v in attributes.items()
            ]:
                path = self.root / f'{name}.bin'
                spool.dump(path)
                self.attributes[name] = Attribute(
                    graph=self,
                    name=name,
                    path=path,
                )

    def render(self, code):
        pass



@weaver.register(name='weave_graph_ingest', unpack=True)
def _weave_graph_ingest(*, index, nodes=None, edges=None):
    index, _can_write = lookup_spool(index)
    assert len(index.columns) == 2, \
        f'The index spool should have two columns, actual: {index.columns!r}'

    attributes = {}
    for what in [nodes, edges]:
        if what is None:
            continue
        
        for name, attribute in what.items():
            attribute, _can_write = lookup_spool(attribute)

            assert len(attribute.columns) == 1, \
                f'The attribute spool should have one column: {name} actual columns: {attribute.columns!r}'

            attributes[name] = attribute

    graph = Graph.create(
        root=Path(__file__).parent.parent / 'graph',
    )

    graph.ingest(index, attributes)

    return {
        'tokens': {
            'fs': graph.fs_token,
            'rw': graph.rw_token,
            'ro': graph.ro_token,
        },
    }


#--- Weave (HTTP Server Interface)

from base64 import (
    urlsafe_b64decode as b64decode,
    urlsafe_b64encode as b64encode,
)

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


def test():
    from string import Template
    import pprint

    global _g_braid_integrator

    _g_braid_integrator = Integrator.from_files(
        ugrd=Path.cwd() / 'data' / 'UGRD-144x73.dat',
        vgrd=Path.cwd() / 'data' / 'VGRD-144x73.dat',
        vvel=Path.cwd() / 'data' / 'VVEL-144x73.dat',
    )

    app.testing = True
    client = app.test_client()

    def weave(code: str, **kwargs) -> Dict:
        url = '/weave/'

        code = Template(code)
        code = code.substitute(**kwargs)
        code = code.encode('utf-8')
        code = b64encode(code)
        code = code.decode('ascii')

        query_string = {
            'code': code,
        }

        response = client.post(
            url,
            query_string=query_string,
        )

        return response.get_json()

    graph = weave('''
return weave_spool_create{ "graph", {src="I"}, {dst="I"} }
    ''')

    print(f'{graph = !r}')

    traces = weave('''
function quantize(options)
  local x = options[1]
  local lo = options.lo
  local hi = options.hi
  local bins = options.bins

  return math.floor(bins * (x - lo) / (hi - lo))
end

local lat = 35.9606
local lng = -83.9207
local prs = 800.0

local seeds = {}
for i = 1, 10 do
    seeds[#seeds+1] = { lat=lat + math.sin(i), lng=lng + math.cos(i), prs=prs }
end

local spool = [===[${graph_tokens_rw}]===]
local traces = weave_braid_trace{ from='2012-01-01', to='2012-01-30', seeds=seeds }
local N = 10

for i = 1, #traces do
  local trace = traces[i]
  local seed = trace.seed
  local points = trace.points

  local prev = nil
  for j = 1, #points do
    local point = points[j]
    local voxel = 0
    voxel = N * voxel + quantize{point.prs, bins=N, lo=  1.0, hi=1000.0}
    voxel = N * voxel + quantize{point.lng, bins=N, lo=  0.0, hi= 360.0}
    voxel = N * voxel + quantize{point.lat, bins=N, lo=-90.0, hi=  90.0}

    if prev ~= nil then
      weave_spool_emit{ spool, src=prev, dst=voxel }
    end

    prev = voxel
  end
end

return traces
''', graph_tokens_rw=graph['tokens']['rw'])

    print('traces =')    
    # import pprint; pprint.pprint(traces)

    index = weave('''
local _voxel_to_node_id = {}
local _next_node_id = 1
function voxel_to_node_id(voxel)
    node_id = _voxel_to_node_id[voxel]
    if node_id == nil then
        node_id = _next_node_id
        _next_node_id = _next_node_id + 1
        _voxel_to_node_id[voxel] = node_id
    end

    return node_id
end

local connections = weave_spool_items{ [===[${graph_tokens_ro}]===] }
local index = weave_spool_create{ "index", {src="I"}, {dst="I"} }
for i = 1, #connections do
    local connection = connections[i]
    local src = voxel_to_node_id{ connection.src }
    local dst = voxel_to_node_id{ connection.dst }

    weave_spool_emit{ index, src=src, dst=dst }
end

return index
''', graph_tokens_ro=graph['tokens']['ro'])

    print('index =')
    pprint.pprint(index)

    graph = weave('''
local index = [===[${index_tokens_ro}]===]
return weave_graph_ingest{ index=index }
''', index_tokens_ro=index['tokens']['ro'])

    print('graph =')
    pprint.pprint(graph)


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=7772)
    parser.add_argument('--debug', action='store_true')
    args = vars(parser.parse_args())

    if args.pop('test'):
        test()

    else:
        main(**args)


if __name__ == '__main__':
    cli()
