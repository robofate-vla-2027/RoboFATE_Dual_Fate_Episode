"""Original offline asset remapping; factory-injected asset manifest and USD API."""
import hashlib
import json

def remap_offline(env):
    folder = ROOT / "deployment/offline_materials"
    manifest = json.loads((folder / "manifest.json").read_text())
    mapping = {entry["url"]: folder / entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        assert hashlib.sha256((folder / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    original = env.scene_manager.reload_env_scene

    def reload(*a, **kw):
        result = original(*a, **kw)
        for prim in omni.usd.get_context().get_stage().Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                asset = next((spec.default for spec in attr.GetPropertyStack()
                              if isinstance(spec.default, Sdf.AssetPath)), None)
                if asset is not None and asset.path.startswith(("http:", "https:", "omniverse:")):
                    assert asset.path in mapping, f"Uncached external asset: {asset.path}"
                    attr.Set(Sdf.AssetPath(str(mapping[asset.path])))
        return result
    env.scene_manager.reload_env_scene = reload
