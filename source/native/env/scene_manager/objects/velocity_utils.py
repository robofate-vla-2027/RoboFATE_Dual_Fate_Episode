"""Reset rigid-body velocities through the CPU/GPU-compatible combined API."""


def apply_default_velocities(prim, view=None):
    linear = prim._default_linear_velocity
    angular = prim._default_angular_velocity
    if linear is None and angular is None:
        return
    if view is None:
        view = prim._rigid_prim_view
    if not view.is_physics_handle_valid():
        prim.set_default_state(linear_velocity=linear, angular_velocity=angular)
        return
    # Preserve unspecified components; separate setters are unsupported on GPU.
    velocities = view.get_velocities()
    if linear is not None:
        velocities[0, :3] = prim._backend_utils.convert(linear, device=prim._device)
    if angular is not None:
        velocities[0, 3:] = prim._backend_utils.convert(angular, device=prim._device)
    view.set_velocities(velocities)
