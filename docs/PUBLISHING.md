# Publishing to GitHub

I prepared this archive as a repository-ready project snapshot. I have not
configured a remote repository in it. My suggested repository name is
`fly-brain-celeste`.

## Repository details

**Description:** Connectome-inspired Celeste reinforcement learning with a
trainable MCNS recurrent graph, a C# Everest bridge, and Python PPO.

**Suggested topics:** `celeste`, `everest`, `reinforcement-learning`, `connectome`,
`python`, `csharp`, `ppo`, `neuroscience`.

I keep `README.md`, `Source/`, and `Python/` at the repository root. Upload the
extracted contents so those paths appear
directly at the repository root. Keep the prebuilt mod and its build metadata
together, including the loose `prebuilt/FlyBrainCeleste.dll`. I use ordinary Git
for the selected graph and keep full dataset exports, the game installation,
and checkpoints local.

## Prepare the first commit

Run the checks in [DEVELOPMENT.md](DEVELOPMENT.md), including:

```powershell
python tools/check_repository.py
python tools/check_release.py
```

If you edit public files, refresh the archive and manifest before committing:

```powershell
python tools/package_release.py
python tools/check_release.py
```

An unchanged C# source can reuse the bundled DLL. C# changes require a fresh
build as described in the development guide. Packaging updates hashes; it does
not run tests or establish gameplay results.

Create an **empty GitHub repository**, then run these commands from this folder.
Replace `YOUR_USERNAME` with your actual GitHub username:

```powershell
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
git commit -m "Add FlyBrain Celeste learning project"
git remote add origin https://github.com/YOUR_USERNAME/fly-brain-celeste.git
git push -u origin main
```

Check staged files before committing. `.gitignore` excludes local settings,
worker copies, training output, saves, environment files, keys, and logs.
Ignored files already tracked in an existing repository remain tracked; inspect
that repository's staged changes when updating it.

GitHub Desktop also works: add this extracted folder as a local repository,
review changes, make the initial commit, and choose Publish repository.

## Validate the published snapshot

Open the repository's Actions tab and inspect **Offline checks**. The workflow
tests Python on Ubuntu and Windows, checks formatting and documentation links,
verifies the release manifest, and runs optional tensor tests on CPU.
It does not download Celeste or exercise native game physics.

After the first push, copy the repository URL into the optional `repository-code`
field in `CITATION.cff` if desired, then repackage before committing that edit.
No username, repository URL, badge result, or published release is assumed here.

## Prepare a downloadable release

1. Run the relevant checks, update [CHANGELOG.md](CHANGELOG.md), and record
   evidence in [VALIDATION.md](VALIDATION.md).
2. Package the checked project and run `tools/check_release.py`.
3. Commit the source, build information, and refreshed hash manifest together.
4. Tag that checked commit with an unused release tag. The software version
   currently remains 4.1.0; this cleanup does not change bridge or model formats.
5. Create a GitHub release from that tag and attach
   `dist/FlyBrainCeleste-LEARN.zip`. The outer archive contains the launcher,
   Python controller, documentation, and nested worker mods.

Use recorded evidence to describe results. Distinguish search-assisted training
progress from frozen-policy gameplay. Add recordings and hardware measurements
when available.
