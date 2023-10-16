from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'spot_rob750'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name + "/launch/"), glob('launch/*_launch*')),
        # (os.path.join('share', package_name + "/models/"), glob('models/spot_description/*.urdf')),
        (os.path.join('share', package_name + "/config/"), glob('config/*')),
        # (os.path.join('share', package_name + "/maps/"), glob('maps/*')),
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
        ],
    },
)
