## PostgresDB

### Tables

- library
    - id
    - path
    - name

- library_prototypes
    - id
    - library:id
    - name
    - real_name
    - locals:jsonb
    - arguments:jsonb
    - num_args
    - heap_arguemnts:jsonb
    - return

- binary
    - id
    - path
    - name
    - optimisation
    - compiler

- symbol
    - binary:id
    - name
    - real_name
    - asm
    - vex
    - analysis
