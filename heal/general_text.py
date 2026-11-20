"""General-knowledge calibration windows (README 6l): text that makes a model recall names, places, dates and numbers.

Nothing is downloaded. A local model (Qwen3.5-9B behind the mlx server) writes about 300 topics listed here, in three
kinds of request, thinking off; the conversations are then cut into 512-token windows in the chat format.

  python heal/general_text.py write --out cache/general/text.jsonl      (needs the server on 127.0.0.1:8080)
  python heal/general_text.py windows --text cache/general/text.jsonl --tokenizer <dir with the 35B tokenizer> \
         --out cache/general/data_general.npz                          -> G (calibration), H (held-out)
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOPICS = """
the Byzantine Empire | the Mongol conquests | the Hanseatic League | the Thirty Years' War | the Meiji era in Japan |
the Haitian Revolution | the unification of Italy | the Ottoman Empire | the Mali Empire | the Inca Empire |
the Tang dynasty | the Mughal Empire | the Spanish Armada | the American Civil War | the Congress of Vienna |
the Russian Revolution of 1917 | the partition of India | the Cuban Missile Crisis | the Suez Crisis | the Apollo programme |
the Black Death | the Reformation | the Peloponnesian War | the Punic Wars | the Achaemenid Empire |
the Viking Age | the Crusades | the Hundred Years' War | the Glorious Revolution | the Opium Wars |
the Taiping Rebellion | the Scramble for Africa | the Boer Wars | the Mexican Revolution | the Spanish Civil War |
the Marshall Plan | the Korean War | the Vietnam War | the fall of apartheid | the European Union's history |
the geography of the Andes | the Sahara | the Amazon basin | the Himalayas | the Great Rift Valley |
the rivers of Siberia | the islands of Indonesia | the fjords of Norway | the Great Barrier Reef | the Caspian Sea |
the countries and capitals of Central Asia | the countries and capitals of West Africa | the countries and capitals of South America |
the countries and capitals of Southeast Asia | the countries and capitals of the Balkans | the countries and capitals of the Caribbean |
the provinces of Canada | the states of Australia | the prefectures of Japan | the regions of France |
the cantons of Switzerland | the deserts of Asia | the great lakes of Africa | the Danube | the Mekong |
the Mississippi River system | volcanoes of the Pacific Ring of Fire | the Arctic Ocean | Antarctica | the Mediterranean Sea |
the periodic table's transition metals | the noble gases | the alkali metals | the lanthanides | the halogens |
the structure of DNA | photosynthesis | the citric acid cycle | the human heart | the human immune system |
the nervous system | the endocrine glands | vitamins and their deficiencies | antibiotics and their discovery | vaccines in history |
plate tectonics | the rock cycle | the geological time scale | mass extinctions | the water cycle |
the planets of the Solar System | the moons of Jupiter and Saturn | comets and asteroids | the life cycle of stars | black holes |
the history of the telescope | exoplanets | the Big Bang theory | the cosmic microwave background | the Hubble Space Telescope |
Newton's laws and their history | thermodynamics and its laws | electromagnetism from Faraday to Maxwell | special relativity | quantum mechanics' founders |
the Standard Model of particle physics | radioactivity and its discoverers | superconductivity | lasers | semiconductors and the transistor |
the classification of animals | the evolution of mammals | dinosaurs | birds of prey | marine mammals |
insects and their orders | the great apes | coral reefs | rainforest ecosystems | the tundra |
famous mathematicians of the 18th century | famous mathematicians of the 19th century | the history of calculus | prime numbers and famous results about them | the history of geometry from Euclid |
the Fields Medal and its winners | famous unsolved and solved conjectures | the history of probability | the history of algebra | the history of zero and numerals |
the Italian Renaissance painters | the Dutch Golden Age painters | Impressionism | Cubism and its artists | Surrealism |
the Baroque composers | the Classical-era composers | the Romantic composers | twentieth-century composers | the history of opera |
jazz and its great musicians | the history of rock music | the Beatles | the history of cinema | the French New Wave |
Japanese cinema and its directors | the Academy Awards' history | Shakespeare's plays | the Russian novelists of the 19th century | the Latin American Boom writers |
the Nobel Prize in Literature and notable laureates | the Greek tragedians | Dante's Divine Comedy | the Arabian Nights | the Tale of Genji |
English Romantic poets | American novelists of the 20th century | African literature in the 20th century | German literature from Goethe to Mann | classical Chinese novels |
Gothic cathedrals | the architecture of ancient Rome | Islamic architecture | the Bauhaus | skyscrapers and their history |
the Seven Wonders of the Ancient World | UNESCO World Heritage sites in Asia | the great museums of Europe | the pyramids of Egypt | Angkor Wat |
the philosophy of ancient Greece | the Enlightenment philosophers | German idealism | existentialism | analytic philosophy |
Confucianism and its texts | Buddhism and its schools | Hinduism's scriptures | the history of Christianity's councils | Islam's early caliphates |
Greek mythology | Norse mythology | Egyptian mythology | the Mahabharata and the Ramayana | Arthurian legend |
the history of printing | the steam engine and the Industrial Revolution | the history of railways | the history of aviation | the history of the automobile |
the history of the computer | the history of the Internet | programming languages and their creators | the history of cryptography | the history of telecommunications |
the Python standard library's main modules | the Unix shell and its common commands | the HTTP protocol and its status codes | SQL and relational databases | the Linux kernel's history |
the history of video games | the history of photography | the history of electricity generation | nuclear power and its history | renewable energy technologies |
the history of medicine from Hippocrates | the discovery of insulin | the history of anaesthesia | the eradication of smallpox | the Human Genome Project |
blood groups and transfusion | the bones of the human skeleton | the major organs and their functions | infectious diseases and their agents | the history of surgery |
the history of money and banking | the Great Depression | the Bretton Woods system | central banks | famous economists and their ideas |
the United Nations and its agencies | the Geneva Conventions | the history of human rights law | constitutions of the world | the International Court of Justice |
the Olympic Games' history | the FIFA World Cup's history | the history of tennis and its champions | the Tour de France | the history of chess and its champions |
sumo and its traditions | cricket and its history | baseball's history | mountaineering and the first ascents | polar exploration |
the Age of Discovery and its navigators | the Silk Road | the history of cartography | famous shipwrecks | the Panama and Suez canals |
the languages of Europe and their families | the writing systems of the world | the history of the English language | the Japanese writing system | the Romance languages |
world cuisines and their signature dishes | the history of tea and coffee | wine regions of the world | spices and the spice trade | cheese varieties and their origins |
units of measurement and the SI system | the calendar and its history | time zones and the prime meridian | famous constants of physics and chemistry | chemical elements named after people and places |
the Nobel Prizes in Physics and notable laureates | the Nobel Prizes in Chemistry and notable laureates | the Nobel Prizes in Medicine and notable laureates | women scientists in history | inventors of the 19th century |
explorers of Africa | the kings and queens of England | the emperors of Rome | the presidents of the United States | the shoguns of Japan |
the tsars of Russia | the pharaohs of Egypt | the caliphs and sultans | leaders of independence movements | famous generals in history |
earthquakes and tsunamis in history | famous volcanic eruptions | hurricanes and typhoons | famines in history | pandemics in history |
national parks of the world | endangered species | the world's largest cities | the world's longest bridges and tunnels | the world's tallest mountains |
currencies of the world | flags and national symbols | national anthems and their composers | airlines and airports of the world | the world's major ports |
minerals and gemstones | metals and alloys | plastics and polymers | fertilisers and the Haber process | dyes and pigments in history |
trees and forests of the temperate zone | cereals and staple crops | domesticated animals and their origins | medicinal plants | fungi
"""
KINDS = [
    "Write an encyclopedia-style overview of {t}. Give the names, places, dates and numbers that matter. About 450 words.",
    "List the 25 most important facts about {t}, one per line, each with a concrete name, place, date or number.",
    "Write 20 quiz questions about {t}, each followed on the next line by its short answer (a name, a place, a year or a number).",
]
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def topics() -> list[str]:
    return [t.strip() for t in TOPICS.replace("\n", " ").split("|") if t.strip()]


def write(args) -> None:
    import requests

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["i"] for l in open(out)} if out.exists() else set()
    jobs = [(i, t, KINDS[i % 3].format(t=t)) for i, t in enumerate(topics()) if i not in done]
    print(f"{len(done)} kept, {len(jobs)} to write", flush=True)
    lock = threading.Lock()

    def one(job):
        i, t, prompt = job
        body = {"model": "default_model", "messages": [{"role": "user", "content": prompt}], "max_tokens": 900, **SAMPLING,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = requests.post(f"{args.base_url}/chat/completions", json=body, timeout=1200)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:
            print(f"{i}: {e!r}", flush=True)
            return False
        if len(text) > 400:
            with lock, open(out, "a") as f:
                f.write(json.dumps({"i": i, "topic": t, "kind": i % 3, "prompt": prompt, "text": text}, ensure_ascii=False) + "\n")
        return True

    with ThreadPoolExecutor(args.concurrency) as pool:
        ok = list(pool.map(one, jobs))
    print("all written" if all(ok) else f"{ok.count(False)} failed; run again", flush=True)


def windows(args) -> None:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = sorted((json.loads(l) for l in open(args.text)), key=lambda r: r["i"])
    random.Random(0).shuffle(rows)
    ids = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["text"].strip()}]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        ids.append(tok(text, add_special_tokens=False)["input_ids"])
    wins, topics_of, i = [], [], 0
    while i < len(ids):  # a window starts at a conversation and runs on into the next ones until it holds 512 tokens
        w, j = [], i
        while len(w) < args.length and j < len(ids):
            w += ids[j]
            j += 1
        if len(w) < args.length:
            break
        wins.append(w[: args.length])
        topics_of.append(rows[i]["topic"])
        i = j
    n_h = args.held_out
    g, h = np.asarray(wins[:-n_h], dtype=np.int64), np.asarray(wins[-n_h:], dtype=np.int64)
    np.savez(args.out, G=g, H=h)
    lens = sorted(len(x) for x in ids)
    print(f"{len(rows)} texts (median {lens[len(lens) // 2]} tokens) -> G {g.shape}, H {h.shape}; wrote {args.out}")
    Path(str(args.out) + ".topics.json").write_text(json.dumps({"G": topics_of[:-n_h], "H": topics_of[-n_h:]}, ensure_ascii=False, indent=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "windows", "count"])
    ap.add_argument("--out")
    ap.add_argument("--text")
    ap.add_argument("--tokenizer")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--held-out", type=int, default=40)
    args = ap.parse_args()
    if args.cmd == "count":
        print(len(topics()), "topics")
    else:
        {"write": write, "windows": windows}[args.cmd](args)


if __name__ == "__main__":
    main()
