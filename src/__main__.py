"""Entrypoint của package: `python3 -m src [args]` chạy huấn luyện ViACaPu.

Ví dụ (cấu hình chính: acoustic + aux energy loss):
  python3 -m src --use_acoustic 1 --gate_bias -2.0 --aux_energy_w 0.5 \
                 --seed 42 --tag ac_aux --exp_dir exp_seed
"""
from .train import main

if __name__ == "__main__":
    main()
