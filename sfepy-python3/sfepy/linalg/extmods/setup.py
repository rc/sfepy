#!/usr/bin/env python

def configuration(parent_package='', top_path=None):
    import os.path as op
    from numpy.distutils.misc_util import Configuration

    from sfepy import Config

    site_config = Config()
    os_flag = {'posix' : 0, 'windows' : 1}

    auto_dir = op.dirname(__file__)
    auto_name = op.split(auto_dir)[-1]
    config = Configuration(auto_name, parent_package, top_path)

    defines = [('__SDIR__', "'\"%s\"'" % auto_dir),
               ('SFEPY_PLATFORM', os_flag[site_config.system()])]
    if '-DDEBUG_FMF' in site_config.debug_flags():
        defines.append(('DEBUG_FMF', None))

    fem_src = ['common_python.c']
    fem_src = [op.join('../../fem/extmods', ii) for ii in fem_src]

    src = ['crcm.pyx', 'rcm.c']
    config.add_extension('crcm',
                         sources=src + fem_src,
                         extra_compile_args=site_config.compile_flags(),
                         extra_link_args=site_config.link_flags(),
                         include_dirs=[auto_dir, '../../fem/extmods'],
                         define_macros=defines)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
