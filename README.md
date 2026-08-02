# Autonomous WhatsApp Multimodal Notification Routing System
## Technical Architecture, Multi-Agent Pipeline & System Implementation

---

## Executive Summary

The **Autonomous WhatsApp Multimodal Notification Router** is a production-grade multi-agent agentic intelligence system designed to process, classify, evaluate security risks, and route incoming WhatsApp communications across text, voice notes, and images.

The system processes incoming communications and outputs decisions into six mandatory schema fields: `message_id`, `action`, `message_type`, `reason`, `confidence`, and `evidence_message_ids`.

---

## Setup & Execution Instructions

### 1. Prerequisites & Environment Setup
Ensure **Python 3.10+** is installed on your system.

Install all required Python dependencies:
```bash
pip install -r requirements.txt
```

#### Main Dependencies (`requirements.txt`):
* `httpx` (Async HTTP requests for LLM APIs)
* `easyocr` (Local deep-learning OCR fallback)
* `pytesseract` (Tesseract OCR engine fallback)
* `pillow` (Image processing)
* `numpy` (Vector calculations)

---

### 2. Environment Variable Configuration
Set the necessary API key environment variables prior to running the system:

#### Linux / macOS:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GROQ_API_KEY="your_groq_api_key_here"
export ASSEMBLYAI_API_KEY="your_assemblyai_api_key_here"
```

#### Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:ASSEMBLYAI_API_KEY="your_assemblyai_api_key_here"
```

#### Windows CMD:
```cmd
set GEMINI_API_KEY="your_gemini_api_key_here"
set GROQ_API_KEY="your_groq_api_key_here"
set ASSEMBLYAI_API_KEY="your_assemblyai_api_key_here"
```

---

### 3. Pipeline Execution
To run the full autonomous multi-agent pipeline over all 110 messages in `dataset/messages.csv`:

```bash
python code/main.py
```

The output decisions will stream directly and be saved to **`dataset/output.csv`**.

---

### 4. Running Perception Pipeline Diagnostic Evaluation
To evaluate vision perception performance across multiple OCR and vision engines:

```bash
python code/test_media_perception.py
```

---

## System Workflow & Sub-Agent Symbol Architecture Chart

```text
========================================================================================================
                                INCOMING MULTIMODAL MESSAGE
                           (Text, Voice Notes .ogg, Images .jpg)
========================================================================================================
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ AGENT 1: MULTIMODAL PERCEPTION ENGINE                                                                │
│ ├─► [Sub-Agent 1.1]: Vision OCR Engine ──────────► (Gemini 3.5 Flash Lite ──► EasyOCR / Tesseract)   │
│ └─► [Sub-Agent 1.2]: Audio Whisper Engine ──────► (AssemblyAI API Speech-to-Text)                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ AGENT 2: HISTORICAL MEMORY & VECTOR EVIDENCE ENGINE                                                  │
│ ├─► [Sub-Agent 2.1]: Context Builder ────────────► (DND Hours, Opt-Out Preferences, Open Ratios)     │
│ └─► [Sub-Agent 2.2]: Vector Similarity Engine ──► (TF-IDF Cosine Similarity Search & Evidence IDs)   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ AGENT 3: SECURITY GUARDRAILS & ROUTER ENGINE                                                         │
│ ├─► [Sub-Agent 3.1]: Security Audit ─────────────► (Regex Prompt Injection & Domain Spoof Check)     │
│ │                                                  ├─► Threat Detected ────► [ACTION: MUTE (Scam)]   │
│ │                                                  └─► Clean Message ──────► Proceed to Sub-Agent 3.2│
│ └─► [Sub-Agent 3.2]: Router LLM Engine ──────────► Priority Tier: Groq ──► Ollama ──► Gemini         │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ AGENT 4: SUPERVISOR & ARBITRATED CONSENSUS ENGINE                                                    │
│ ├─► [Sub-Agent 4.1]: Consensus Checker ──────────► (Compares Vector Precedent vs Router Decision)   │
│ │                                                  ├─► Direct Agreement ──► Pass 0 Consensus         │
│ │                                                  └─► Disagreement ──────► Proceed to Sub-Agent 4.2│
│ ├─► [Sub-Agent 4.2]: Arbiter Feedback Loop ──────► (3-Pass Iterative Critique Re-injection)        │
│ └─► [Sub-Agent 4.3]: Output Schema Validator ────► (Sanitizes 6-Column Output Dictionary)            │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DYNAMIC BAYESIAN POSTERIOR CONFIDENCE CALIBRATION ENGINE                                              │
│ P(Decision | Evidence) = (P_init * M_consensus * L_vector) / Normalizer   [Capped at max 0.95]        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
========================================================================================================
                                   OUTPUT AUDIT CSV FILE
                dataset/output.csv (6 Mandatory Columns: message_id, action,
                 message_type, reason, confidence, evidence_message_ids)
========================================================================================================
```

---

## 1. Sub-Agent Breakdown & Operational Specifications

### Agent 1: Multimodal Perception Engine (`code/agent1_perception.py`)
- **Sub-Agent 1.1 (Vision OCR Engine)**:
  - Base64 converts `.jpg` image bytes and dispatches to **Gemini 3.5 Flash Lite** (`gemini-3.5-flash-lite`).
  - Retries on the exact current image if API errors occur.
  - Switches to a deep-learning local fallback (**EasyOCR + Tesseract**) if cloud retries are exhausted.
  - Returns strict structured output:
    ```text
    [EXTRACTED_TEXT]: <Verbatim text extracted from image/poster>
    [SCENE_DESCRIPTION]: <1-2 sentence visual description of scene context>
    ```
- **Sub-Agent 1.2 (Audio Whisper Engine)**:
  - Dispatches `.ogg` / `.wav` voice notes to AssemblyAI (`whisper-large-v3` / `best` transcriber) for verbatim speech-to-text.

### Agent 2: Memory & Vector Evidence Engine (`code/agent2_memory_rag.py`)
- **Sub-Agent 2.1 (Context Builder)**:
  - Assembles profile features including user DND hours, `allows_promotions` flags, historical open rates (`past_opened_ratio`), and past dismissal counts.
- **Sub-Agent 2.2 (Vector Similarity Engine)**:
  - Builds a TF-IDF vector space model across past user interaction history (`history.csv`).
  - Calculates continuous Cosine Similarity metrics ($\text{top\_cosine\_sim} \in [0.0, 1.0]$) to retrieve matching evidence IDs (`evidence_message_ids`).

### Agent 3: Security & Router Engine (`code/agent3_router_security.py`)
- **Sub-Agent 3.1 (Security Guardrails Audit)**:
  - Intercepts adversarial prompt injections (e.g., `ignore previous instructions`, `set action=notify`), domain mismatch spoofing (official business domain $\neq$ sender domain), and unverified business credential harvesting before LLM invocation.
- **Sub-Agent 3.2 (Router LLM Engine)**:
  - Dispatches text prompts following sticky priority: **Groq (`llama-3.3-70b-versatile`) $\rightarrow$ Ollama (`minimax-m3:cloud`) $\rightarrow$ Gemini (`gemini-2.5-flash`)**.
  - Evaluates action (`notify`, `digest`, `mute`) and message type.

### Agent 4: Supervisor & Arbitrated Consensus Engine (`code/agent4_supervisor.py`)
- **Sub-Agent 4.1 (Consensus Checker)**:
  - Compares the Router decision against historical memory precedent hints.
- **Sub-Agent 4.2 (Arbiter Feedback Loop)**:
  - When sub-agents disagree, an Arbiter sub-agent inspects the rationale, generates corrective feedback, and re-injects the critique back into the Router LLM across up to 3 iterative passes until consensus is achieved.
- **Sub-Agent 4.3 (Schema Validator & Bayesian Calibration)**:
  - Calculates dynamic Bayesian confidence and formats the final row dict to strictly match the 6-column output specification.

---

## 2. Dynamic Bayesian Posterior Confidence Calibration Engine

To prevent arbitrary hardcoding while maintaining mathematical accuracy, decision confidence is calculated using **Bayes' Theorem for Dynamic Posterior Probability Calibration**:

$$\text{Posterior Confidence} = \min\left(0.95, \, \frac{P(\text{Decision}) \cdot M_{\text{Consensus}} \cdot L_{\text{Vector}}}{P(\text{Decision}) \cdot M_{\text{Consensus}} \cdot L_{\text{Vector}} + (1 - P(\text{Decision}))}\right)$$

### Mathematical Calibration Factors:
1. **$P(\text{Decision})$ (Model Prior Certainty)**: Initial confidence float evaluated for the message decision ($0.05 \le P \le 0.95$).
2. **$M_{\text{Consensus}}$ (Multi-Agent Agreement Likelihood)**:
   - Deterministic Security Rule Match $\rightarrow 1.05$
   - Direct Consensus (Pass 0) $\rightarrow 1.02$
   - Resolved on Pass 1 $\rightarrow 0.92$
   - Resolved on Pass 2 $\rightarrow 0.82$
   - Heavy 3-Pass Arbitration $\rightarrow 0.70$
3. **$L_{\text{Vector}}$ (Evidence Vector Match Likelihood)**:
   - Scaled continuously from vector TF-IDF Cosine Similarity ($1.0 + 0.30 \times \text{top\_cosine\_sim}$).
4. **Pass 0 Confidence Cap**:
   - Strictly capped at **0.95 (95%)** to represent calibrated empirical certainty.

---

## 3. Key Strengths, Robust Edge Case Handling & Fallback Architectures

### A. Robust Edge Case Handling
1. **DND Midnight Boundary Wraparound**: Correctly handles quiet hour ranges spanning across midnight (e.g. `22:00-07:00`) by splitting time into relative minute offsets.
2. **Transactional Order Delivery Exception**: Verified business delivery notices (Amazon, Grab, HDFC) override `allows_promotions=No` user preferences to prevent missing active shipments.
3. **Mass Forward Chain Mitigation**: Automatically mutes viral forward chains with `forwarded_count >= 5` even if @mentions are embedded.
4. **Domain Spoof Mitigation**: Intercepts fake business senders whose email/URL domains do not match official registered domains.

### B. Comprehensive Fallback Architectures & Resilience
1. **Multimodal Vision Fallback Chain**:
   - Primary: **Gemini 3.5 Flash Lite** (`gemini-3.5-flash-lite`) with Base64 image payload.
   - Retries: Up to 3 automatic retries on the current image if transient API errors occur.
   - Secondary Fallback: Deep-learning local OCR combining **EasyOCR** and **Pytesseract** directly on the image if API tiers fail.
2. **LLM Tier Failover Chain**:
   - Primary: **Groq (`llama-3.3-70b-versatile`)**.
   - Secondary: **Ollama (`minimax-m3:cloud`)**.
   - Tertiary: **Gemini (`gemini-2.5-flash`)**.
   - If an active LLM tier experiences rate limiting or connection failure mid-execution, the pipeline automatically switches to the next available tier without losing message state.
3. **JSON Response Auto-Repair**:
   - Handles malformed LLM responses, markdown code block wrappers (```json ... ```), and trailing commas using regex JSON extractors and defensive dictionary fallbacks.
4. **Rate Limit Concurrency & Sleep Backoff**:
   - Utilizes `_api_rate_lock` with `API_CALL_DELAY_SECONDS = 1.5`s delay to maintain high processing throughput while respecting provider RPM ceilings.
