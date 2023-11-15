from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'spot_safe_rl'
submodules = 'spot_safe_rl/submodules'
setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name + "/launch/"), glob('launch/*_launch*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lass6230',
    maintainer_email='due6230@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safe_rl_node = spot_safe_rl.safe_rl_node:main',
        ],
    },
)
