using Celeste.Mod;

namespace CelesteRL.Lockstep;

/// Everest entry point. All behaviour lives in LockstepDriver.
public class LockstepModule : EverestModule {
    public override void Load() => LockstepDriver.Load();

    public override void Unload() => LockstepDriver.Unload();
}
