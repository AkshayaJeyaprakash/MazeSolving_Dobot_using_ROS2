from setuptools import find_packages, setup

package_name = 'maze_solver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aj',
    maintainer_email='aj@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'perception_node = maze_solver.perception_node:main',
        'executor_node = maze_solver.executor_node:main',
        'claude_bridge_node = maze_solver.claude_bridge_node:main'
        ],
    },
)
