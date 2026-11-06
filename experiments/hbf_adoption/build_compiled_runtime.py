"""Rebuild the optional untyped compiled event runtime in a fresh directory."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from run_validation import ROOT,HERE,execute,fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out.resolve();out.mkdir(parents=True,exist_ok=False)
    import Cython
    if Cython.__version__!='3.2.1':raise ValueError('pinned Cython 3.2.1 required')
    inputs=[]
    for name in ('projection_resources.py','projection_engine.py'):
        source=HERE/name;shutil.copy2(source,out/name);inputs.append(fingerprint(source))
    recipe="""from setuptools import setup,Extension
from Cython.Build import cythonize
setup(name='hbf-exact-event-runtime',ext_modules=cythonize(
 [Extension(name,[name+'.py']) for name in ('projection_resources','projection_engine')],
 compiler_directives={'language_level':3,'annotation_typing':False,'infer_types':False,'binding':True,'initializedcheck':True},
 annotate=False))
"""
    (out/'setup.py').write_text(recipe)
    receipt=execute([sys.executable,'setup.py','build_ext','--inplace'],out,out/'build-execution',
        inputs=[out/'setup.py',out/'projection_resources.py',out/'projection_engine.py'])
    if receipt['status']!='pass':raise ValueError('compile failed')
    (out/'manifest.json').write_text(json.dumps(dict(status='completed',sources=inputs,
        recipe=fingerprint(out/'setup.py'),binaries=[fingerprint(p) for p in sorted(out.glob('*.so'))],
        receipt=fingerprint(out/'build-execution/receipt.json')) ,indent=2)+'\n')


if __name__=='__main__':main()
