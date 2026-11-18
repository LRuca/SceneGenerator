"""Agent-owned motion semantics and an explicit guarded fitting entry point.

The agent supplies classifications; this module never infers them from fit loss.
Unknown decisions require review before apply. Legacy scene.fit remains available
for historical experiments and does not imply these checks were performed.
"""
import copy
import numpy as np
from .storage import read_json, write_json, reference, new_run
from .scene import load_scene, validate_scene

MOTIONS = {'static': 'static', 'rigid_dynamic': 'rigid', 'deformable': 'deformable'}


def validate(decisions, scene):
    rows = decisions.get('entities', [])
    ids = [e['id'] for e in scene['entities']]
    if not isinstance(decisions.get('actor'), str) or not decisions['actor'].strip():
        raise ValueError('Explicit agent actor required')
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError('Motion decision must cover every entity exactly once')
    if any(not isinstance(x, dict) for x in rows):
        raise ValueError('Invalid motion decision row')
    row_ids = [x.get('id') for x in rows]
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(ids):
        raise ValueError('Motion decision IDs must be unique and cover the scene')
    for row in rows:
        if row.get('motion') not in {*MOTIONS, 'unknown'}:
            raise ValueError('Unsupported motion decision')
        if not isinstance(row.get('evidence'), (str, list)) or not row['evidence']:
            raise ValueError('Evidence required')
    return decisions


def resolve_scene(scene, decisions, n):
    """Pure conversion, with no file writes until every decision is valid."""
    validate(decisions, scene)
    if any(row['motion'] == 'unknown' for row in decisions['entities']):
        raise ValueError('Unknown motion requires an explicit agent-reviewed branch before fitting')
    result = copy.deepcopy(scene)
    by_id = {x['id']: x for x in decisions['entities']}
    for e in result['entities']:
        row = by_id[e['id']]
        target = MOTIONS[row['motion']]
        if (e['motion'] == 'deformable') != (target == 'deformable'):
            raise ValueError('Changing deformation topology requires a representation revision: ' + e['id'])
        poses = np.asarray(e['entity_to_world'], float)
        e['motion'] = target
        e['agent_motion_decision'] = copy.deepcopy(row)
        e['motion_source'] = 'agent_semantic_decision'
        e['motion_model_locked'] = True
        constraints = copy.deepcopy(e.get('fit_constraints', {}))
        constraints['locked_motion'] = target
        if target == 'static':
            # This is an initialization only; the shared pose is fitted afterward.
            e['entity_to_world'] = [poses[len(poses) // 2].tolist()]
            constraints['lock_yaw'] = True
            e['motion_initialization'] = 'middle source pose; shared pose subsequently fitted'
        elif target == 'rigid' and len(poses) == 1:
            e['entity_to_world'] = np.repeat(poses, n, axis=0).tolist()
        e['fit_constraints'] = constraints
        # Old curves/state must never override the new motion parameterization.
        for key in ['continuous_motion', 'exists', 'observation_available', 'observation_used',
                    'state_source', 'visibility_state', 'visible_fraction_estimate',
                    'uncertainty_proxy', 'evidence_summary', 'existence_source']:
            e.pop(key, None)
        e['renderable'] = [True] * n if target == 'static' else list(e.get('observed', e['renderable']))
    for key in ['persistence', 'state_semantics']:
        result.pop(key, None)
    validate_scene(result, n)
    return result


def apply(scene_path, decisions_path, out):
    scene, _, a = load_scene(scene_path)
    dec = read_json(decisions_path)
    if dec.get('source_scene') != reference(scene_path):
        raise ValueError('Motion decisions must bind the exact source scene hash')
    result = resolve_scene(scene, dec, len(a['depth']))
    result['agent_motion_decisions'] = reference(decisions_path)
    result['agent_motion_source_scene'] = reference(scene_path)
    p = new_run(out)
    write_json(p / 'scene.json', result)
    write_json(p / 'report.json', {
        'schema': 'scene_baseline.agent_motion_decision.v1', 'actor': dec['actor'],
        'decisions': dec, 'all_motion_labels_locked': True,
        'source_scene': reference(scene_path), 'source_decisions': reference(decisions_path),
        'limitations': ['Rigid trajectories still use legacy smoothness regularization, not hard speed/acceleration bounds',
                        'Unknown branches are not automatically generated or selected',
                        'Shared static pose is a hypothesis, not proof of stationarity']})
    return result


def require_locked(scene):
    rows = []
    for e in scene['entities']:
        row = e.get('agent_motion_decision')
        if not row or row.get('motion') not in MOTIONS or not e.get('motion_model_locked'):
            raise ValueError('Explicit resolved agent motion decision required: ' + e['id'])
        if MOTIONS[row['motion']] != e['motion'] or e.get('fit_constraints', {}).get('locked_motion') != e['motion']:
            raise ValueError('Agent motion semantics changed: ' + e['id'])
        rows.append(row)
    if not rows:
        raise ValueError('Empty scene')


def fit_locked(scene_path, out, **kwargs):
    from .fit_scene import fit
    scene, _, _ = load_scene(scene_path)
    require_locked(scene)
    if any(e['motion'] == 'deformable' for e in scene['entities']):
        raise ValueError('Use the dedicated deformation fitter after motion review')
    report = fit(scene_path, out, **kwargs)
    fitted, _, _ = load_scene(out / 'scene.json')
    require_locked(fitted)
    return report
