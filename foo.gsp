#pragma gs shader(positional)
void main() {
    float x = sin(gs_NodeIndex / 100.0);
    float y = cos(gs_NodeIndex / 100.0);

    gs_NodePosition = vec3(x, y, 0.);
}

#pragma gs shader(relational)
void main() {
}

#pragma gs shader(appearance)
void main() {
    gs_FragColor = vec4(0.1);
}