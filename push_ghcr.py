#!/usr/bin/env python3
"""Push an OCI image layout to ghcr.io, blob by blob, with the gh CLI's login. Prints no secret.

usage: push_ghcr.py <layout dir> <name, e.g. mvizn/cudaopencv> <tag>

<layout dir> is an unpacked `docker save` archive (`mkdir L && tar -xf cudaopencv-oci.tar -C L`). If its index.json
points at an image index (a multi-platform or attestation index), the linux/amd64 image manifest in it is pushed.
Needs `gh auth login` with a token that may write packages (scope write:packages) and curl. Credentials: the gh
token goes to curl through its config on stdin (`curl -K -`), the registry token through a 0600 header file that
is deleted right after each request; neither ever appears in a command line. Pushing does not change the package's
visibility; a new package is created private."""
import json, os, subprocess, sys
layout, name, tag = sys.argv[1:4]
R = f'https://ghcr.io/v2/{name}'
user = subprocess.check_output(['gh', 'api', 'user', '--jq', '.login']).decode().strip()
def token():
    pat = subprocess.check_output(['gh', 'auth', 'token']).decode().strip()
    # Basic auth through curl's config on stdin, not on the command line.
    cfg = f'user = "{user}:{pat}"\nurl = "https://ghcr.io/token?service=ghcr.io&scope=repository:{name}:pull,push"\n'
    out = subprocess.run(['curl', '-s', '-K', '-'], input=cfg.encode(), capture_output=True, check=True).stdout
    return json.loads(out)['token']
def curl(args, tok):
    # The token goes through a 0600 file, never on the command line (a process list would show it).
    import tempfile
    fd, hf = tempfile.mkstemp(prefix='ghcr-hdr-')
    try:
        os.fchmod(fd, 0o600); os.write(fd, f'Authorization: Bearer {tok}\n'.encode()); os.close(fd)
        return subprocess.run(['curl', '-s', '-H', f'@{hf}'] + args, capture_output=True)
    finally:
        os.unlink(hf)
idx = json.load(open(f'{layout}/index.json'))['manifests'][0]
mdig = idx['digest'].split(':')[1]
m = json.loads(open(f'{layout}/blobs/sha256/{mdig}', 'rb').read())
if 'manifests' in m:   # an image index: take its linux/amd64 image manifest
    mdig = [d for d in m['manifests'] if d.get('platform', {}).get('architecture') == 'amd64'
            and d.get('platform', {}).get('os') == 'linux'][0]['digest'].split(':')[1]
    m = json.loads(open(f'{layout}/blobs/sha256/{mdig}', 'rb').read())
print('manifest', 'sha256:' + mdig, m['mediaType'], flush=True)
for d in [m['config']] + m['layers']:
    dig = d['digest']; path = f"{layout}/blobs/sha256/{dig.split(':')[1]}"
    tok = token()
    code = curl(['-o', '/dev/null', '-w', '%{http_code}', '-I', f'{R}/blobs/{dig}'], tok).stdout.decode()
    if code == '200':
        print('exists  ', dig[:19], d['size'], flush=True); continue
    hdr = curl(['-D', '-', '-o', '/dev/null', '-X', 'POST', f'{R}/blobs/uploads/'], tok).stdout.decode()
    loc = [l.split(':', 1)[1].strip() for l in hdr.splitlines() if l.lower().startswith('location:')][0]
    if loc.startswith('/'): loc = 'https://ghcr.io' + loc
    sep = '&' if '?' in loc else '?'
    r = curl(['-o', '/dev/null', '-w', '%{http_code}', '-X', 'PUT', '-H', 'Content-Type: application/octet-stream',
              '-T', path, f'{loc}{sep}digest={dig}'], tok)
    code = r.stdout.decode()
    print('uploaded' if code == '201' else f'FAILED {code}', dig[:19], d['size'], flush=True)
    if code != '201': sys.exit(1)
tok = token()
r = curl(['-o', '/dev/null', '-w', '%{http_code}', '-X', 'PUT', '-H', f"Content-Type: {m['mediaType']}",
          '--data-binary', f'@{layout}/blobs/sha256/{mdig}', f'{R}/manifests/{tag}'], tok)
print('manifest PUT', r.stdout.decode(), flush=True)
h = curl(['-I', '-H', f"Accept: {m['mediaType']}", f'{R}/manifests/{tag}'], tok).stdout.decode()
print([l.strip() for l in h.splitlines() if l.lower().startswith(('http/', 'docker-content-digest'))], flush=True)
