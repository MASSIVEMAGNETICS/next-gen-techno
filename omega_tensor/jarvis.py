"""
J.A.R.V.I.S. - Just A Rather Very Intelligent System
Built on Omega Tensor's NextGenTransformer + GravitronAttention Architecture

"At your service, sir."
"""

import time
import numpy as np

from .tensor import Tensor
from .autograd import no_grad
from .revolutionary_models import NextGenTransformer
from . import nn

# ---------------------------------------------------------------------------
# Character-level tokeniser constants
# ---------------------------------------------------------------------------
_VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\"()-/\n"
_CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(_VOCAB_CHARS)}  # 0 = padding
VOCAB_SIZE = len(_VOCAB_CHARS) + 1
MAX_SEQ_LEN = 128


# ---------------------------------------------------------------------------
# Neural core
# ---------------------------------------------------------------------------
class JARVISBrain(nn.Module):
    """
    JARVIS's neural processing core.

    Uses a NextGenTransformer with GravitronAttention to classify the intent
    of natural-language input.  Keyword-based boosting on top of the raw
    logits produces robust, hybrid rule-neural classification without the
    need for pre-training data.
    """

    INTENTS = [
        "greeting",
        "status",
        "compute",
        "search",
        "time",
        "system",
        "help",
        "farewell",
        "compliment",
        "unknown",
    ]

    _INTENT_KEYWORDS = {
        "greeting":   ["hello", "hi", "hey", "greetings", "good morning", "good evening"],
        "status":     ["status", "report", "update", "situation", "condition"],
        "compute":    ["calculate", "compute", "solve", "math", "add", "multiply", "divide"],
        "search":     ["search", "find", "look up", "locate", "where", "who", "what is"],
        "time":       ["time", "date", "clock", "hour", "minute", "today", "now"],
        "system":     ["system", "diagnostics", "cpu", "memory", "power", "reactor"],
        "help":       ["help", "assist", "capabilities", "can you", "what can"],
        "farewell":   ["goodbye", "bye", "shutdown", "exit", "quit", "sleep"],
        "compliment": ["good job", "well done", "great", "excellent", "nice work", "perfect"],
        "unknown":    [],
    }

    def __init__(self):
        super().__init__()
        self.transformer = NextGenTransformer(
            num_embeddings=VOCAB_SIZE,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            feedforward_dim=128,
            num_classes=len(self.INTENTS),
        )

    def forward(self, x):
        return self.transformer(x)

    @staticmethod
    def tokenize(text):
        """Encode *text* as a fixed-length padded integer sequence."""
        text = text.lower()[:MAX_SEQ_LEN]
        tokens = [_CHAR_TO_IDX.get(c, 0) for c in text]
        tokens += [0] * (MAX_SEQ_LEN - len(tokens))
        return tokens[:MAX_SEQ_LEN]

    def _keyword_boost(self, text):
        """Return a per-intent bonus derived from simple keyword matching."""
        text_lower = text.lower()
        boost = np.zeros(len(self.INTENTS), dtype=np.float32)
        for i, intent in enumerate(self.INTENTS):
            for kw in self._INTENT_KEYWORDS.get(intent, []):
                if kw in text_lower:
                    boost[i] += 10.0
        return boost

    def classify(self, text):
        """
        Predict the intent label for *text*.

        Combines transformer logits with keyword-based boosting so that the
        assistant works well even without fine-tuning.
        """
        tokens = self.tokenize(text)
        x = Tensor(np.array([tokens], dtype=np.float32))
        with no_grad():
            logits = self.forward(x)
        combined = logits.data[0] + self._keyword_boost(text)
        return self.INTENTS[int(np.argmax(combined))]


# ---------------------------------------------------------------------------
# JARVIS assistant
# ---------------------------------------------------------------------------
class JARVIS:
    """
    J.A.R.V.I.S. – Just A Rather Very Intelligent System.

    An always-on, always-listening AI assistant powered by Omega Tensor's
    NextGenTransformer architecture with GravitronAttention.

    Examples
    --------
    Single-turn usage::

        j = JARVIS()
        print(j.respond("Hello JARVIS"))

    Always-on interactive mode (reads stdin until an exit command)::

        j = JARVIS()
        j.run()
    """

    _BANNER = (
        "\n"
        "  ___  _  ____   _  __ ___  ____  \n"
        " |   /_\\|  _ \\ \\ / //_ _|/ ___| \n"
        "  \\ / _ \\ |_) | \\ / / | | \\___ \\ \n"
        "   V/_/ \\_\\____/  \\_/ |___||____/ \n"
        "\n"
        "  Just A Rather Very Intelligent System\n"
        "  Powered by Omega Tensor · GravitronAttention Core\n"
        "  ─────────────────────────────────────────────────\n"
        '  "At your service, sir."\n'
    )

    _RESPONSES = {
        "greeting": [
            "Good day, sir. All systems nominal.",
            "At your service. How may I assist you today?",
            "Hello. The Arc Reactor is running at full capacity.",
            "Greetings. I have been waiting. Shall we begin?",
        ],
        "status": [
            "All systems fully operational. Power output stable at 4 gigajoules.",
            "Status report: GravitronAttention core running at peak efficiency. No anomalies detected.",
            "Systems nominal. Neural pathways optimized. Awaiting further instruction.",
            "All clear, sir. Weapons offline, as per your standing order.",
        ],
        "compute": [
            "Running tensor calculations now. The answer is… elegant.",
            "Neural computation engaged. Processing at maximum throughput.",
            "Omega Tensor arithmetic cores fully engaged. Solution incoming.",
            "Computation complete. Accuracy: god-tier.",
        ],
        "search": [
            "Accessing global information matrix. One moment, sir.",
            "Cross-referencing all available data sources.",
            "Search initiated. I shall have results before your coffee cools.",
            "Querying decentralized knowledge nodes now.",
        ],
        "time": [
            "The current time is {time}.",
            "It is {time_ampm} on {date}.",
            "Stardate {stardate}. Local time: {time}.",
        ],
        "system": [
            "Arc Reactor output: 4 GJ/s. GravitronAttention core: online. All sensors: nominal.",
            "Omega Tensor runtime: healthy. Decentralized tensor registry: active.",
            "Diagnostics complete. No critical faults. Recommend continued operation.",
            "System integrity: 100%. SpaceTimeEmbedding grid: calibrated.",
        ],
        "help": [
            "I can assist with: status reports, calculations, information retrieval, "
            "time queries, system diagnostics, and general conversation.",
            "My capabilities include: real-time tensor inference, GravitronAttention-powered NLU, "
            "and making you feel like Tony Stark.",
            "Commands I understand: greetings, status, compute, search, time, system, help, farewell.",
        ],
        "farewell": [
            "Powering down non-essential systems. Goodbye, sir.",
            "As you wish. I shall be here when you return.",
            "Initiating standby mode. It has been a pleasure.",
            "Until next time. JARVIS signing off.",
        ],
        "compliment": [
            "Thank you, sir. I do try.",
            "Flattery will get you everywhere, sir.",
            "I am merely operating as designed. You built me well.",
            "Much appreciated. I shall file that under 'rare praise'.",
        ],
        "unknown": [
            "I am not entirely sure I follow, sir. Could you rephrase?",
            "My intent classifier is stumped. That does not happen often.",
            "Fascinating. I have logged this for further analysis.",
            "I believe that requires clarification. Or more coffee.",
        ],
    }

    def __init__(self, owner="Sir", verbose=True):
        """
        Parameters
        ----------
        owner : str
            Name used when addressing the user (e.g. "Mr Stark").
        verbose : bool
            Whether to print the banner when starting interactive mode.
        """
        self.owner = owner
        self.verbose = verbose
        self.brain = JARVISBrain()
        self._response_counters = {k: 0 for k in self._RESPONSES}
        self.running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(self, text):
        """
        Process *text* and return JARVIS's response string.

        Parameters
        ----------
        text : str
            Natural-language input from the user.

        Returns
        -------
        str
            JARVIS response.
        """
        if not text.strip():
            return "Awaiting input, {}.".format(self.owner)

        intent = self.brain.classify(text)
        response = self._pick_response(intent)

        # Personalise response with owner name on greetings / farewells or randomly
        if intent in ("greeting", "farewell") or np.random.random() < 0.25:
            # Strip trailing punctuation cleanly before appending the owner name
            if response and response[-1] in ".!?":
                response = response[:-1]
            response = "{}, {}.".format(response, self.owner)

        return response

    def run(self, prompt="You: ", exit_commands=("quit", "exit", "shutdown jarvis", "bye")):
        """
        Start JARVIS in always-on interactive mode.

        Reads from stdin indefinitely until an exit command is received or
        an EOF / KeyboardInterrupt is detected.

        Parameters
        ----------
        prompt : str
            The input prompt shown to the user.
        exit_commands : tuple[str, ...]
            Phrases that trigger a graceful shutdown.
        """
        self.running = True
        if self.verbose:
            print(self._BANNER)
            print("JARVIS: Good day, {}. Always-on mode activated.".format(self.owner))
            print("        (type 'quit' or 'shutdown jarvis' to exit)\n")

        while self.running:
            try:
                user_input = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nJARVIS: Emergency shutdown detected. Goodbye, {}.".format(self.owner))
                self.running = False
                break

            if user_input.strip().lower() in exit_commands:
                print("JARVIS:", self._pick_response("farewell"))
                self.running = False
                break

            response = self.respond(user_input)
            print("JARVIS:", response)
            print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_response(self, intent):
        """Round-robin selection so JARVIS does not repeat himself."""
        options = self._RESPONSES.get(intent, self._RESPONSES["unknown"])
        # Initialise counter on demand for any intent not pre-seeded (e.g. dynamic intents)
        counter = self._response_counters.get(intent, 0)
        raw = options[counter % len(options)]
        self._response_counters[intent] = counter + 1
        return self._render_time_tokens(raw)

    @staticmethod
    def _render_time_tokens(text):
        """Replace time-format placeholders with live values."""
        now = time.localtime()
        return (
            text
            .replace("{time}", time.strftime("%H:%M:%S", now))
            .replace("{time_ampm}", time.strftime("%I:%M %p", now))
            .replace("{date}", time.strftime("%A, %B %d, %Y", now))
            .replace("{stardate}", time.strftime("%Y.%m%d", now))
        )
