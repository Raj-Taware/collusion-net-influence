from netinf.llm import CachedGemini


def test_cache_replays_without_api_call(tmp_path, monkeypatch):
    llm = CachedGemini(model="fake-model", cache_path=tmp_path / "c.sqlite")
    calls = []
    monkeypatch.setattr(llm, "_call", lambda req: calls.append(req) or f"reply-{req['seed']}")

    assert llm.generate("hi", seed=1) == "reply-1"
    assert llm.generate("hi", seed=1) == "reply-1"
    assert llm.generate("hi", seed=2) == "reply-2"
    assert len(calls) == 2
    assert llm.usage.cache_hits == 1

    # cache persists across instances
    llm2 = CachedGemini(model="fake-model", cache_path=tmp_path / "c.sqlite")
    monkeypatch.setattr(llm2, "_call", lambda req: (_ for _ in ()).throw(AssertionError("should hit cache")))
    assert llm2.generate("hi", seed=2) == "reply-2"
