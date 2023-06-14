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

go---docker() {
    pexec "${self:?}" docker \
    exec "${self:?}" "$@"
}

go---gs() {
    pexec "${self:?}" gs \
    exec "${self:?}" "$@"
}

go-exec() {
    pexec "$@"
}

go-Test() {
    pexec "${self:?}" docker \
    exec "${self:?}" gs \
    exec "${self:?}" virtualenv \
    exec python3 -m weaver \
        --test \
        ##
}

Server_bind=127.0.0.1

go-Server() {
    "${FUNCNAME[0]:?}-$@"
}

Server_a_bind=${Server_bind:?}
Server_a_port=7002
Server_a_host=a.vaas.is.mediocreatbest.xyz

go-Server-a() {
    pexec python3 -m weaver \
        --bind "${Server_a_bind:?}" \
        --port "${Server_a_port:?}" \
        --host "${Server_a_host:?}" \
        ##
}

Server_b_bind=${Server_bind:?}
Server_b_port=7003
Server_b_host=b.vaas.is.mediocreatbest.xyz

go-Server-b() {
    pexec python3 -m weaver \
        --bind "${Server_b_bind:?}" \
        --port "${Server_b_port:?}" \
        --host "${Server_b_host:?}" \
        ##
}

Server_c_bind=${Server_bind:?}
Server_c_port=7004
Server_c_host=c.vaas.is.mediocreatbest.xyz

go-Server-c() {
    pexec python3 -m weaver \
        --bind "${Server_c_bind:?}" \
        --port "${Server_c_port:?}" \
        --host "${Server_c_host:?}" \
        ##
}

Server_d_bind=${Server_bind:?}
Server_d_port=7001
Server_d_host=d.vaas.is.mediocreatbest.xyz

go-Server-d() {
    pexec "${self:?}" docker \
    exec "${self:?}" gs \
    exec "${self:?}" virtualenv \
    exec python3 -m weaver \
        --bind "${Server_d_bind:?}" \
        --port "${Server_d_port:?}" \
        --host "${Server_d_host:?}" \
        ##
}

Server_e_bind=${Server_bind:?}
Server_e_port=7002
Server_e_host=e.vaas.is.mediocreatbest.xyz

go-Server-e() {
    pexec python3 -m weaver \
        --bind "${Server_e_bind:?}" \
        --port "${Server_e_port:?}" \
        --host "${Server_e_host:?}" \
        ##
}

Server_f_bind=${Server_bind:?}
Server_f_port=7003
Server_f_host=f.vaas.is.mediocreatbest.xyz

go-Server-f() {
    pexec python3 -m weaver \
        --bind "${Server_f_bind:?}" \
        --port "${Server_f_port:?}" \
        --host "${Server_f_host:?}" \
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


#--- Docker

go-docker() {
    pexec "${root:?}/external/GraphShaders/go.sh" docker \
        "$@" \
        ##
}


#--- GraphShaders (gs)

go-gs() {
    pexec "${root:?}/external/GraphShaders/go.sh" gs \
        "$@" \
        ##
}


#---

virtualenv_path=${root:?}/venv
virtualenv_python=python3
virtualenv_install=(
    lupa
    numpy
    scipy
    flask
    flask-cors
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