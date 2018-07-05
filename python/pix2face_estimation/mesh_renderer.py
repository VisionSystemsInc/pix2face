import face3d

_mesh_renderer = None


def get_mesh_renderer():
    global _mesh_renderer
    # get the mesh renderer, which is created only once, lazily
    if _mesh_renderer is None:
        _mesh_renderer = face3d.mesh_renderer()
    return _mesh_renderer
