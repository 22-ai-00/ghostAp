"""Regression coverage for task-derived Spec roles across non-coding work."""

from types import SimpleNamespace

from src.spec_engine.artifacts import parse_spec_artifact
from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks, _has_valid_required_experts
from src.spec_engine.models import ReviewContext, SpecArtifact, SpecCycle, SpecExpertRole, SpecPhase, SpecWorkItem
from src.spec_engine.prompts import build_build_prompt, build_plan_prompt, build_spec_prompt, build_task_prompt
from src.spec_engine.reporter import SpecReporter
from src.spec_engine.review import _plan_roles
from src.spec_engine.review_artifacts import ReviewArtifacts


def _settings():
    return SimpleNamespace(spec_review_total_roles_max=8)


def _context(artifacts):
    return ReviewContext(
        cycle=1,
        session=None,
        settings=_settings(),
        project=None,
        send_prompt_with_retry_fn=lambda *_args, **_kwargs: None,
        build_review_exception_diagnostics_fn=lambda *_args, **_kwargs: {},
        circuit=None,
        artifacts=artifacts,
    )


def test_writing_spec_preserves_inferred_ideation_reader_editor_and_reviewer_roles():
    artifact, errors = parse_spec_artifact(
        """```json
        {
          "goals": ["publish an accessible history essay"],
          "functional_spec": ["explain the subject clearly"],
          "non_functional_requirements": ["accurate"],
          "acceptance_criteria": ["a general reader can follow the argument"],
          "out_of_scope": [], "risks": [], "clarification_questions": [], "decisions": [],
          "required_experts": [
            {"role": "Ideation editor", "purpose": "test the angle and narrative promise", "focus": ["angle"], "checks": ["the premise is original and useful"]},
            {"role": "Target reader", "purpose": "test comprehension for the intended audience", "focus": ["clarity"], "checks": ["a general reader can follow it"]},
            {"role": "Developmental editor", "purpose": "test structure and prose", "focus": ["structure"], "checks": ["the argument has a coherent arc"]},
            {"role": "Fact reviewer", "purpose": "test claims and evidence", "focus": ["sources"], "checks": ["material claims are supported"]}
          ]
        }
        ```"""
    )

    assert artifact is not None
    assert not errors
    assert [expert.role for expert in artifact.required_experts] == [
        "Ideation editor", "Target reader", "Developmental editor", "Fact reviewer"
    ]

    context = _context(
        ReviewArtifacts(
            cycle_number=1,
            requirement="Write an accessible history essay",
            cwd="/tmp",
            required_experts=[expert.to_dict() for expert in artifact.required_experts],
        )
    )
    roles, _ = _plan_roles(context)

    assert [role.display_name for role in roles] == [
        "Ideation editor", "Target reader", "Developmental editor", "Fact reviewer", "完成度与方向把控"
    ]
    assert all(role.category == "task_derived" for role in roles[:-1])
    assert all(role.role_id.startswith("expert_") for role in roles[:-1])


def test_inferred_domain_experts_are_used_without_keyword_domain_templates():
    context = _context(
        ReviewArtifacts(
            cycle_number=1,
            requirement="Create a consent process for a clinical study",
            cwd="/tmp",
            required_experts=[
                {"role": "Clinical ethics specialist", "purpose": "protect participant autonomy", "focus": ["consent"], "checks": ["risks are understandable"]},
                {"role": "Study participant advocate", "purpose": "represent participant concerns", "focus": ["burden"], "checks": ["choices are voluntary"]},
            ],
        )
    )

    roles, _ = _plan_roles(context)

    assert [role.display_name for role in roles] == [
        "Clinical ethics specialist", "Study participant advocate", "完成度与方向把控"
    ]
    assert {role.mission for role in roles[:-1]} == {
        "protect participant autonomy", "represent participant concerns"
    }


def test_non_coding_phase_prompts_keep_the_task_domain_and_require_expert_inference():
    spec_prompt = build_spec_prompt("Write a museum audio-guide script", "/tmp/museum", "", "")
    plan_prompt = build_plan_prompt("museum script", "/tmp/museum")
    task_prompt = build_task_prompt("museum script")
    build_prompt = build_build_prompt([], "museum script", "/tmp/museum", "")

    assert "required_experts" in spec_prompt
    assert "软件架构师" not in spec_prompt
    assert "必须在 required_experts 中包含创意/选题、目标读者、编辑、审稿/事实或论证核查四种独立视角" in spec_prompt
    assert "不要假定这是软件工程" in plan_prompt
    assert "研究、创作、设计、运营或实现工作" in task_prompt
    assert "不要把非编码任务改写成代码实现" in build_prompt
    assert SpecPhase.BUILD.display_name == "执行交付"
    assert SpecReporter()._extract_phase_summary(SpecPhase.BUILD, "delivered\nartifact") == "执行输出 2 行"


def test_new_spec_without_experts_is_repaired_once_before_planning():
    initial = "not valid JSON"
    repaired = """```json
    {"goals":["write"],"functional_spec":[],"non_functional_requirements":[],
    "acceptance_criteria":["reader understands"],"out_of_scope":[],"risks":[],
    "clarification_questions":[],"decisions":[],"required_experts":[
      {"role":"Reader","purpose":"test comprehension","focus":["clarity"],"checks":["a reader understands"]},
      {"role":"Editor","purpose":"test structure","focus":["structure"],"checks":["the guide flows"]},
      {"role":"Fact reviewer","purpose":"test claims","focus":["facts"],"checks":["claims are supported"]}
    ]}
    ```"""
    engine = SpecEngine(chat_id="adaptive-roles", root_path="/tmp")
    prompts: list[str] = []
    responses = iter((initial, repaired))
    engine._run_phase = lambda _cycle, _phase, prompt, _callbacks, _timeout, **_kwargs: (prompts.append(prompt), next(responses))[1]
    engine._finish_phase = lambda *_args, **_kwargs: None
    cycle = SpecCycle(cycle_number=1)

    output = engine._run_spec_phase(1, cycle, "Write a guide", None, SpecEngineCallbacks(), 30)

    assert output == repaired
    assert len(prompts) == 2
    assert "required_experts" in prompts[1]
    assert "Write a guide" in prompts[1]
    assert "/tmp" in prompts[1]
    assert [expert.role for expert in cycle.spec_artifact.required_experts] == ["Reader", "Editor", "Fact reviewer"]


def test_loaded_work_item_without_roles_is_repaired_before_the_single_spec_callback(tmp_path):
    stored_spec = tmp_path / "legacy-spec.json"
    stored_spec.write_text(
        '{"goals":["improve"],"functional_spec":[],"non_functional_requirements":[],'
        '"acceptance_criteria":["done"],"out_of_scope":[],"risks":[],'
        '"clarification_questions":[],"decisions":[]}',
        encoding="utf-8",
    )
    repaired = """```json
    {"goals":["improve"],"functional_spec":[],"non_functional_requirements":[],
    "acceptance_criteria":["done"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],
    "required_experts":[
      {"role":"Domain expert","purpose":"validate the improvement","focus":["domain"],"checks":["constraints hold"]},
      {"role":"Audience representative","purpose":"validate usability","focus":["audience"],"checks":["need is met"]},
      {"role":"Evidence reviewer","purpose":"validate proof","focus":["evidence"],"checks":["criterion is proven"]}
    ]}
    ```"""
    engine = SpecEngine(chat_id="loaded-spec", root_path=str(tmp_path))
    prompts: list[str] = []
    started: list[SpecCycle] = []
    delivered: list[str] = []

    def repair_phase(_cycle, phase, prompt, callbacks, _timeout, **_kwargs):
        prompts.append(prompt)
        callbacks.on_phase_start(_cycle, phase)
        return repaired

    engine._run_phase = repair_phase
    engine._finish_phase = lambda *_args, **_kwargs: None
    cycle = SpecCycle(cycle_number=2)
    work_item = SpecWorkItem(item_id="Q-1", question="Improve onboarding", created_in_cycle=1, spec_path=str(stored_spec))

    output = engine._run_spec_phase(
        2,
        cycle,
        "Original project objective",
        work_item,
        SpecEngineCallbacks(
            on_phase_start=lambda _cycle, _phase: started.append(cycle),
            on_phase_done=lambda _cycle, _phase, text: delivered.append(text),
        ),
        30,
    )

    assert output == repaired
    assert len(prompts) == 1
    assert "Original project objective" in prompts[0]
    assert stored_spec.read_text(encoding="utf-8") in prompts[0]
    assert started == [cycle]
    assert delivered == [repaired]
    assert [expert.role for expert in cycle.spec_artifact.required_experts] == [
        "Domain expert", "Audience representative", "Evidence reviewer"
    ]


def test_valid_loaded_work_item_emits_spec_start_before_done(tmp_path):
    stored_spec = tmp_path / "valid-spec.json"
    stored_spec.write_text(
        '{"goals":["improve"],"functional_spec":[],"non_functional_requirements":[],'
        '"acceptance_criteria":["done"],"out_of_scope":[],"risks":[],'
        '"clarification_questions":[],"decisions":[],"required_experts":['
        '{"role":"Domain","purpose":"validate","focus":["domain"],"checks":["holds"]},'
        '{"role":"Audience","purpose":"use","focus":["audience"],"checks":["works"]},'
        '{"role":"Evidence","purpose":"prove","focus":["evidence"],"checks":["shown"]}]}',
        encoding="utf-8",
    )
    engine = SpecEngine(chat_id="valid-loaded", root_path=str(tmp_path))
    engine._run_phase = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("valid loaded spec must not prompt"))
    engine._finish_phase = lambda *_args, **_kwargs: None
    events: list[str] = []

    engine._run_spec_phase(
        1,
        SpecCycle(cycle_number=1),
        "Objective",
        SpecWorkItem(item_id="Q-2", question="Improve", created_in_cycle=1, spec_path=str(stored_spec)),
        SpecEngineCallbacks(
            on_phase_start=lambda *_args: events.append("start"),
            on_phase_done=lambda *_args: events.append("done"),
        ),
        30,
    )

    assert events == ["start", "done"]


def test_new_role_plan_rejects_duplicate_or_incomplete_experts():
    duplicate, _ = parse_spec_artifact(
        """```json
        {"goals":[],"functional_spec":[],"non_functional_requirements":[],"acceptance_criteria":["done"],
        "out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"required_experts":[
          {"role":"Reader","purpose":"read","focus":["clarity"],"checks":["understands"]},
          {"role":" reader ","purpose":"duplicate","focus":["clarity"],"checks":["understands"]},
          {"role":"Editor","purpose":"edit","focus":[],"checks":["flows"]}
        ]}
        ```"""
    )

    assert duplicate is not None
    assert not _has_valid_required_experts(duplicate)

    over_limit = SpecArtifact(required_experts=[
        SpecExpertRole(f"Expert {index}", "review", ["focus"], ["check"])
        for index in range(8)
    ])
    assert not _has_valid_required_experts(over_limit)
