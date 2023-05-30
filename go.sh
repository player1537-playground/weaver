#!/usr/bin/env bash
die() { printf $'Error: %s\n' "$*" >&2; exit 1; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
self=${root:?}/${BASH_SOURCE[0]##*/}
project=${root##*/}
pexec() { >&2 printf exec; >&2 printf ' %q' "$@"; >&2 printf '\n'; exec "$@"; }
#---

go---virtualenv() {
    pexec "${self:?}" virtualenv \
    exec "${self:?}" "$@"
}

Server_app=weaver
Server_bind=127.0.0.1
Server_port=7001
Server_host=vaas.is.mediocreatbest.xyz

go-Server() {
    pexec python3 -m "${Server_app:?}" \
        --bind "${Server_bind:?}" \
        --port "${Server_port:?}" \
        --host "${Server_host:?}" \
        ##
}

go-Request() {
    local url
    url=${1:?need url}

    exec < <(pexec python3 -m base64 \
        ##
    )

    exec < <(pexec curl \
        --location \
        --silent \
        --get \
        -X POST \
        "${url:?}" \
        --data-urlencode code@- \
        ##
    )

    pexec cat

    pexec python3 -m json.tool \
        ##
}

Integrate_url=http://${Server_host:?}:${Server_port:?}/weave/

go-Integrate() {
    pexec "${self:?}" Request \
        "${Integrate_url:?}" \
        <<'EOF'
return braid{ from='2012-01-01', to='2012-06-30', seeds={
    { lat=35.9606, lng=83.9207, prs=800.0 },
}}
EOF
}

Spool_url=https://${Server_host:?}/weave/

go-Spool() {
    pexec "${self:?}" Request \
        "${Spool_url:?}" \
        <<'EOF'
local spool = create('testing', 'ff')
print(spool)

emit(spool, 1.23, 3.45)
emit(spool, 2.23, 6.45)
emit(spool, 4.23, 9.45)
emit(spool, 8.23, 1.45)

return spool.tokens.ro
EOF
}


#---

virtualenv_path=${root:?}/venv
virtualenv_python=python3
virtualenv_install=(
    lupa
    numpy
    scipy
    flask
)

go-virtualenv() {
    "${FUNCNAME[0]:?}-$@"
}

go-virtualenv-create() {
    pexec "${virtualenv_python:?}" -m venv \
        "${virtualenv_path:?}" \
        ##
}

go-virtualenv-install() {
    pexec "${virtualenv_path:?}/bin/python" -m pip \
        install \
        "${virtualenv_install[@]}" \
        ##
}

go-virtualenv-exec() {
    source "${virtualenv_path:?}/bin/activate" && \
    pexec "$@"
}


#---
test -f "${root:?}/env.sh" && source "${_:?}"
go-"$@"