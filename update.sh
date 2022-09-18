rm -r build/*
rm -r dist/*
rm -r wheelhouse/*
python setup.py sdist bdist_wheel --plat-name=manylinux1_x86_64
#auditwheel repair dist/REVEALER-0.0.*-cp39-cp39-linux_x86_64.whl
#cp wheelhouse/REVEALER-0.0.*-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl dist
#cp wheelhouse/REVEALER-0.0.*-cp39-cp39-manylinux*.whl dist
#rm dist/REVEALER-0.0.*-cp39-cp39-linux_x86_64.whl
twine upload --repository testpypi dist/*