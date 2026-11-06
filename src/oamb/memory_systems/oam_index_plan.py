"""Read-only operator proof of the configured OAM dense discovery query plan."""

from __future__ import annotations

import json
import subprocess
from typing import Any

# Runs inside the selected OAM image. No product checkout is imported by OAMB.
_PROBE = r"""
import hashlib,json,os,re,sys,time
import psycopg
from psycopg import sql
from oam.auth.credentials import AuthorizedUser
from oam.config import load_config
from oam.db.vector_indexes import owner_index_name
from oam.retrieval.repository import RecallRepository
config_path,tenant,user,mode=sys.argv[1:]
config=load_config(config_path).config
assert config.database.vector_index_mode==mode
class RecordedRepository(RecallRepository):
    def _execute(self,query,parameters=(),*,configure_ann=False):
        rows=super()._execute(query,parameters,configure_ann=configure_ann)
        if configure_ann:
            values={**self.time_parameters,**parameters,'ann_tenant':tenant,'ann_user':user}
            prepared=self.connection.execute(
                'SELECT name FROM pg_prepared_statements ORDER BY prepare_time DESC LIMIT 1'
            ).fetchone()
            if prepared:
                keys=dict.fromkeys(re.findall(r'%\(([^)]+)\)s',query))
                statement=sql.SQL('EXPLAIN (ANALYZE,FORMAT JSON) EXECUTE {}({})').format(
                    sql.Identifier(prepared[0]),sql.SQL(',').join(sql.Literal(values[k]) for k in keys))
                self.plan=self.connection.execute(statement).fetchone()[0][0]['Plan']
            else:
                self.plan=self.connection.execute('EXPLAIN (ANALYZE,FORMAT JSON) '+query,values).fetchone()[0][0]['Plan']
            self.query_sha256=hashlib.sha256(query.encode()).hexdigest()
        return rows
def indexes(node):
    result=[node['Index Name']] if 'Index Name' in node else []
    return result+[name for child in node.get('Plans',[]) for name in indexes(child)]
with psycopg.connect(os.environ[config.database.migration_dsn_env]) as connection:
    vector=connection.execute('SELECT embedding::text FROM search_units WHERE tenant_id=%s '
        'AND user_id=%s AND embedding IS NOT NULL ORDER BY unit_id LIMIT 1',(tenant,user)).fetchone()
    assert vector is not None,'owner has no embedded search units'
    connection.execute('SET LOCAL ROLE oam_content')
    authority=AuthorizedUser('benchmark-probe',tenant,'benchmark-probe',user,frozenset({'memory:read'}))
    repo=RecordedRepository(connection,authority,config,deadline=time.monotonic()+60)
    rows=repo.dense(json.loads(vector[0]),5)
    expected=owner_index_name(tenant,user) if mode=='large' else 'search_units_embedding'
    names=indexes(repo.plan)
    assert rows,'dense query produced no eligible evidence'
    assert expected in names,'expected vector index was not selected'
    print(json.dumps({'tenant_id':tenant,'user_id':user,'mode':mode,'expected_index':expected,
        'selected_indexes':names,'query_sha256':repo.query_sha256,'returned_units':[r['unit_id'] for r in rows],
        'plan':repo.plan}))
"""


def capture_index_plan(
    container: str, config: str, tenant: str, user: str, mode: str
) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "exec", container, "python", "-c", _PROBE, config, tenant, user, mode],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "OAM dense query plan verification failed; no index-selection claim is valid"
        )
    report: dict[str, Any] = json.loads(result.stdout)
    if report.get("expected_index") not in report.get("selected_indexes", []):
        raise ValueError("expected vector index missing from actual dense query plan")
    return report
