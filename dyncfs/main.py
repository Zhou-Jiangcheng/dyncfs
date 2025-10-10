import argparse

from .configuration import CfsConfig
from .cfs_dynamic import (
    create_dynamic_lib,
    compute_dynamic_cfs_sequential,
    compute_dynamic_cfs_parallel,
    compute_dynamic_cfs_fix_depth_sequential,
    compute_dynamic_cfs_fix_depth_parallel,
    run_all_dynamic,
)
from .cfs_static import (
    create_static_lib,
    compute_static_cfs,
    compute_static_cfs_fix_depth,
    run_all_static,
)


def main():
    parser = argparse.ArgumentParser(description="dyncfs command line tool")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )

    # static
    parser.add_argument(
        "--create-static-lib", action="store_true", help="Create static stress library"
    )
    parser.add_argument(
        "--compute-static-cfs",
        action="store_true",
        help="Compute static dCFS on obs faults",
    )
    parser.add_argument(
        "--compute-static-cfs-fix-depth",
        action="store_true",
        help="Compute static dCFS at fixed depth",
    )
    parser.add_argument(
        "--run-static",
        action="store_true",
        help="Create static stress library and "
        "Compute static dCFS on obs faults and "
        "Compute static dCFS at fixed depth",
    )

    # dynamic
    parser.add_argument(
        "--create-dynamic-lib",
        action="store_true",
        help="Create dynamic stress library",
    )
    parser.add_argument(
        "--compute-dynamic-cfs",
        action="store_true",
        help="Compute dynamic dCFS on obs faults",
    )
    parser.add_argument(
        "--compute-dynamic-cfs-fix-depth",
        action="store_true",
        help="Compute dynamic dCFS at fixed depth",
    )
    parser.add_argument(
        "--run-dynamic",
        action="store_true",
        help="Create dynamic stress library and "
        "Compute dynamic dCFS on obs faults and "
        "Compute dynamic dCFS at fixed depth",
    )

    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Create static and dynamic stress library and "
        "Compute static and dynamic dCFS",
    )

    args = parser.parse_args()

    print(f"Using configuration file: {args.config}")
    config = CfsConfig()
    config.read_config(args.config)

    if args.create_static_lib:
        create_static_lib(config)
    if args.compute_static_cfs:
        compute_static_cfs(config)
    if args.compute_static_cfs_fix_depth:
        compute_static_cfs_fix_depth(config)
    if args.run_static:
        run_all_static(config)

    if args.create_dynamic_lib:
        create_dynamic_lib(config)
    if args.compute_dynamic_cfs:
        if config.processes_num == 1:
            compute_dynamic_cfs_sequential(config)
        else:
            compute_dynamic_cfs_parallel(config)
    if args.compute_dynamic_cfs_fix_depth:
        if config.processes_num == 1:
            compute_dynamic_cfs_fix_depth_sequential(config)
        else:
            compute_dynamic_cfs_fix_depth_parallel(config)
    if args.run_dynamic:
        run_all_dynamic(config)

    if args.run_all:
        run_all_static(config)
        run_all_dynamic(config)


if __name__ == "__main__":
    main()
