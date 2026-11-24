"""Original exact Conveyor dependency checks with external paths/pins."""
import tomllib
NAME = 'isaacsim.asset.gen.conveyor'
TASK = 'pick_from_conveyor_by_image'

def required(context):
    extensions = context.get('native_kit_extensions', [])
    expected = [NAME] if context['task'] == TASK else []
    need(type(extensions) is list and extensions == expected, 'Exact reviewed task extension set required')
    if expected:
        need(context['native_total_steps'] == 700 and type(context['native_total_steps']) is int,
                    'Conveyor native budget mismatch')
    return expected


def verify(context):
    names = required(context)
    if not names:
        return None
    pin = context['native_extension_pins']
    need(pin['path'] == PIN['path'], 'Unexpected extension pin set')
    need(pin == PIN, 'Extension pin differs from execution policy')
    data = decode(read_ref(pin))
    need(data['schema'] == 'robofate.native-extension.v1' and data['extension'] == NAME
                and data['version'] == '1.0.23' and data['task'] == TASK and data['native_total_steps'] == 700,
                'Installed extension/task version differs')
    paths = [ROOT/'config/extension.toml', ROOT/'bin/libisaacsim.asset.gen.conveyor.plugin.so']
    need([r['path'] for r in data['files']] == [str(p) for p in paths], 'Exact installed extension files required')
    raws = [read_ref(r, expected_path=p) for r,p in zip(data['files'],paths)]
    package = tomllib.loads(raws[0].decode())
    need(package['package']['version'] == '1.0.23'
                and NAME in [m['name'] for m in package['python']['module']]
                and package['native']['plugin'] == [{'path':'bin/*.plugin','recursive':False}],
                'Installed native-node declaration differs')
    need(raws[1][:4] == b'\x7fELF', 'Expected installed Linux native plugin')
    return pin
