#!/usr/bin/env bash
set -euo pipefail

export PGPASSWORD='p04a65caa469ca22db1726db2288c67c85cc7fea05f781a91368da5cb0315d08f'
export PGSSLMODE=require

psql -h cbhnv71uilek74.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com -U u6a28i7le0nnij -d db4p0t7ikc2hof <<'SQL'
\set ON_ERROR_STOP on
TRUNCATE TABLE intubation_record, "user", alembic_version CASCADE;
\copy "user" (id,email,name,password_hash,created_at) FROM '/mnt/c/Users/Massimo.Giacon/intuB/db_export/user.csv' WITH (FORMAT csv, HEADER true);
\copy intubation_record (id,operator_id,age,weight,height,sex,dtm,dii,mallampati,stop_bang,alganzouri,drug_used,technique,success,cormack,difficult_binary,created_at) FROM '/mnt/c/Users/Massimo.Giacon/intuB/db_export/intubation_record.csv' WITH (FORMAT csv, HEADER true);
\copy alembic_version (version_num) FROM '/mnt/c/Users/Massimo.Giacon/intuB/db_export/alembic_version.csv' WITH (FORMAT csv, HEADER true);
SQL

psql -h cbhnv71uilek74.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com -U u6a28i7le0nnij -d db4p0t7ikc2hof <<'SQL'
select (select count(*) from "user") as users,
       (select count(*) from intubation_record) as records,
       (select count(*) from alembic_version) as alembic;
SQL