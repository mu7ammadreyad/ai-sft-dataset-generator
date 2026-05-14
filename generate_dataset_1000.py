# ==============================================================================
#  مولّد داتا التدريب — 1000 مثال | 10 لكل موضوع | 100 موضوع
#  مُحسَّن لتدريب نموذج 600M | حد أقصى 1024 كلمة لكل مثال
#  دعم مفاتيح Gemini متعددة + حفظ تلقائي + استئناف من نقطة التوقف
# ==============================================================================

from google import genai
import json
import time
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ==============================================================================
#  ⚙️  الإعدادات — عدّلها فقط هنا
# ==============================================================================

GEMINI_API_KEYS = [
    "AIzaSyDPRDKlbdIWtrIq_hdnhotvzX-LoZSAbVg",
    "YOUR_GEMINI_API_KEY_2",
    "YOUR_GEMINI_API_KEY_3",
]

MODEL_ID          = "gemma-4-26b-a4b-it"
OUTPUT_FILE       = "training_dataset_1000.json"
CHECKPOINT_FILE   = "checkpoint.json"       # استئناف تلقائي عند الانقطاع
FAILED_FILE       = "failed_tasks.json"     # المهام الفاشلة للمراجعة
SAMPLES_PER_TOPIC = 10                      # عدد العينات لكل موضوع
MAX_WORDS         = 900                     # حد الكلمات (أقل من 1024 للأمان)
DELAY_BETWEEN_CALLS = 1.5                   # ثانية بين كل API call
SAVE_EVERY        = 5                      # حفظ تلقائي كل N عينة

# ==============================================================================
#  📋  الـ 100 موضوع — 10 أقسام × 10 مواضيع
# ==============================================================================

TOPICS = [
    # ── 1. ترحيب وتعارف (5 مواضيع) ──────────────────────────────────────────
    {"id": 1,  "section": "ترحيب وتعارف",   "topic": "تحية المستخدم والرد عليه بأسلوب دافئ"},
    {"id": 2,  "section": "ترحيب وتعارف",   "topic": "التعريف بالمساعد الذكي وإمكاناته"},
    {"id": 3,  "section": "ترحيب وتعارف",   "topic": "الحديث عن المزاج والمشاعر اليومية"},
    {"id": 4,  "section": "ترحيب وتعارف",   "topic": "شكر المستخدم وتوديعه بأسلوب لطيف"},
    {"id": 5,  "section": "ترحيب وتعارف",   "topic": "الاستفسار عن احتياجات المستخدم وتوجيهه"},

    # ── 2. صحة جسدية (10 مواضيع) ─────────────────────────────────────────────
    {"id": 6,  "section": "صحة جسدية",      "topic": "نصائح لتحسين جودة النوم"},
    {"id": 7,  "section": "صحة جسدية",      "topic": "أعراض نقص الفيتامينات الشائعة"},
    {"id": 8,  "section": "صحة جسدية",      "topic": "التغذية السليمة والوجبات المتوازنة"},
    {"id": 9,  "section": "صحة جسدية",      "topic": "الرياضة المنزلية للمبتدئين"},
    {"id": 10, "section": "صحة جسدية",      "topic": "الإسعافات الأولية الأساسية"},
    {"id": 11, "section": "صحة جسدية",      "topic": "صحة العين عند الاستخدام المطول للشاشات"},
    {"id": 12, "section": "صحة جسدية",      "topic": "الوقاية من نزلات البرد والإنفلونزا"},
    {"id": 13, "section": "صحة جسدية",      "topic": "صحة الأسنان واللثة"},
    {"id": 14, "section": "صحة جسدية",      "topic": "شرب الماء وأهميته للجسم"},
    {"id": 15, "section": "صحة جسدية",      "topic": "آلام الظهر والعمود الفقري عند الجلوس الطويل"},

    # ── 3. صحة نفسية (10 مواضيع) ─────────────────────────────────────────────
    {"id": 16, "section": "صحة نفسية",      "topic": "التعامل مع التوتر والقلق اليومي"},
    {"id": 17, "section": "صحة نفسية",      "topic": "تحسين المزاج في الأيام الصعبة"},
    {"id": 18, "section": "صحة نفسية",      "topic": "الشعور بالوحدة والعزلة"},
    {"id": 19, "section": "صحة نفسية",      "topic": "التعامل مع الحزن والإحباط"},
    {"id": 20, "section": "صحة نفسية",      "topic": "بناء الثقة بالنفس"},
    {"id": 21, "section": "صحة نفسية",      "topic": "التفكير السلبي وكيفية التغلب عليه"},
    {"id": 22, "section": "صحة نفسية",      "topic": "الاسترخاء والتنفس العميق"},
    {"id": 23, "section": "صحة نفسية",      "topic": "إدارة الغضب"},
    {"id": 24, "section": "صحة نفسية",      "topic": "الضغط العصبي في العمل والدراسة"},
    {"id": 25, "section": "صحة نفسية",      "topic": "التوازن بين الحياة الشخصية والعمل"},

    # ── 4. تقنية مبسطة (10 مواضيع) ───────────────────────────────────────────
    {"id": 26, "section": "تقنية مبسطة",    "topic": "كيف يعمل الإنترنت بشكل مبسط"},
    {"id": 27, "section": "تقنية مبسطة",    "topic": "ما هو الذكاء الاصطناعي وكيف يعمل"},
    {"id": 28, "section": "تقنية مبسطة",    "topic": "الفرق بين RAM و ROM والتخزين"},
    {"id": 29, "section": "تقنية مبسطة",    "topic": "ما هو الـ Cloud وفوائده"},
    {"id": 30, "section": "تقنية مبسطة",    "topic": "أمان كلمات المرور والحسابات الإلكترونية"},
    {"id": 31, "section": "تقنية مبسطة",    "topic": "ما هو الـ VPN ولماذا يستخدمه الناس"},
    {"id": 32, "section": "تقنية مبسطة",    "topic": "الفرق بين Android و iOS"},
    {"id": 33, "section": "تقنية مبسطة",    "topic": "كيف تعمل محركات البحث"},
    {"id": 34, "section": "تقنية مبسطة",    "topic": "حماية الخصوصية الرقمية"},
    {"id": 35, "section": "تقنية مبسطة",    "topic": "التصيد الإلكتروني والاحتيال الرقمي"},

    # ── 5. لغة عربية (10 مواضيع) ─────────────────────────────────────────────
    {"id": 36, "section": "لغة عربية",      "topic": "تصحيح الأخطاء النحوية الشائعة"},
    {"id": 37, "section": "لغة عربية",      "topic": "شرح معاني الكلمات الصعبة"},
    {"id": 38, "section": "لغة عربية",      "topic": "الفرق بين كلمتين متشابهتين في المعنى"},
    {"id": 39, "section": "لغة عربية",      "topic": "قواعد الإعراب الأساسية"},
    {"id": 40, "section": "لغة عربية",      "topic": "علامات الترقيم واستخدامها"},
    {"id": 41, "section": "لغة عربية",      "topic": "كيفية كتابة موضوع تعبير أو مقال"},
    {"id": 42, "section": "لغة عربية",      "topic": "الأساليب البلاغية والمجاز"},
    {"id": 43, "section": "لغة عربية",      "topic": "كتابة همزة الوصل والقطع"},
    {"id": 44, "section": "لغة عربية",      "topic": "المثنى والجمع وصيغ المبالغة"},
    {"id": 45, "section": "لغة عربية",      "topic": "تحسين أسلوب الكتابة العربية"},

    # ── 6. تعليم ودراسة (10 مواضيع) ──────────────────────────────────────────
    {"id": 46, "section": "تعليم ودراسة",   "topic": "تقنيات الحفظ الفعّال"},
    {"id": 47, "section": "تعليم ودراسة",   "topic": "الاستعداد للامتحانات"},
    {"id": 48, "section": "تعليم ودراسة",   "topic": "التعلم الذاتي عبر الإنترنت"},
    {"id": 49, "section": "تعليم ودراسة",   "topic": "كيفية البحث العلمي الصحيح"},
    {"id": 50, "section": "تعليم ودراسة",   "topic": "قراءة الكتب بفاعلية واستيعاب"},
    {"id": 51, "section": "تعليم ودراسة",   "topic": "تعلم لغة أجنبية جديدة"},
    {"id": 52, "section": "تعليم ودراسة",   "topic": "فهم المواد الدراسية الصعبة"},
    {"id": 53, "section": "تعليم ودراسة",   "topic": "أخذ الملاحظات بكفاءة"},
    {"id": 54, "section": "تعليم ودراسة",   "topic": "التوازن بين الدراسة والحياة"},
    {"id": 55, "section": "تعليم ودراسة",   "topic": "اختيار التخصص الجامعي المناسب"},

    # ── 7. إنتاجية وعمل (10 مواضيع) ──────────────────────────────────────────
    {"id": 56, "section": "إنتاجية وعمل",   "topic": "تقنية Pomodoro لإدارة الوقت"},
    {"id": 57, "section": "إنتاجية وعمل",   "topic": "تحديد الأولويات وإدارة المهام"},
    {"id": 58, "section": "إنتاجية وعمل",   "topic": "التخلص من التسويف والمماطلة"},
    {"id": 59, "section": "إنتاجية وعمل",   "topic": "التخطيط اليومي والأسبوعي"},
    {"id": 60, "section": "إنتاجية وعمل",   "topic": "التركيز في العمل وتجنب المشتتات"},
    {"id": 61, "section": "إنتاجية وعمل",   "topic": "بناء عادات إيجابية جديدة"},
    {"id": 62, "section": "إنتاجية وعمل",   "topic": "العمل الحر Freelance للمبتدئين"},
    {"id": 63, "section": "إنتاجية وعمل",   "topic": "كتابة السيرة الذاتية باحترافية"},
    {"id": 64, "section": "إنتاجية وعمل",   "topic": "مهارات المقابلة الوظيفية"},
    {"id": 65, "section": "إنتاجية وعمل",   "topic": "الفرق بين العاجل والمهم في العمل"},

    # ── 8. مهارات شخصية (10 مواضيع) ──────────────────────────────────────────
    {"id": 66, "section": "مهارات شخصية",   "topic": "مهارات التواصل والتعبير عن النفس"},
    {"id": 67, "section": "مهارات شخصية",   "topic": "فن الإقناع والتأثير في الآخرين"},
    {"id": 68, "section": "مهارات شخصية",   "topic": "حل النزاعات والخلافات"},
    {"id": 69, "section": "مهارات شخصية",   "topic": "الذكاء العاطفي وفهم المشاعر"},
    {"id": 70, "section": "مهارات شخصية",   "topic": "مهارات العمل ضمن فريق"},
    {"id": 71, "section": "مهارات شخصية",   "topic": "مهارات القيادة وإدارة الآخرين"},
    {"id": 72, "section": "مهارات شخصية",   "topic": "فن الاستماع الفعّال"},
    {"id": 73, "section": "مهارات شخصية",   "topic": "تقديم النقد البنّاء وتلقّيه"},
    {"id": 74, "section": "مهارات شخصية",   "topic": "بناء علاقات اجتماعية صحية"},
    {"id": 75, "section": "مهارات شخصية",   "topic": "التعامل مع الشخصيات الصعبة"},

    # ── 9. مالية شخصية (10 مواضيع) ───────────────────────────────────────────
    {"id": 76, "section": "مالية شخصية",    "topic": "أساسيات الادخار وبناء صندوق طوارئ"},
    {"id": 77, "section": "مالية شخصية",    "topic": "التخطيط المالي الشخصي الشهري"},
    {"id": 78, "section": "مالية شخصية",    "topic": "الفرق بين الأصول والخصوم"},
    {"id": 79, "section": "مالية شخصية",    "topic": "مفهوم التضخم وتأثيره على المدخرات"},
    {"id": 80, "section": "مالية شخصية",    "topic": "كيفية فتح مشروع صغير بميزانية محدودة"},
    {"id": 81, "section": "مالية شخصية",    "topic": "التسويق الإلكتروني للمبتدئين"},
    {"id": 82, "section": "مالية شخصية",    "topic": "إدارة الديون والقروض"},
    {"id": 83, "section": "مالية شخصية",    "topic": "التسوق الذكي وتجنب الإسراف"},
    {"id": 84, "section": "مالية شخصية",    "topic": "الاستثمار للمبتدئين بمفاهيم مبسطة"},
    {"id": 85, "section": "مالية شخصية",    "topic": "توفير المصاريف اليومية"},

    # ── 10. حياة يومية (10 مواضيع + علوم 5) ──────────────────────────────────
    {"id": 86, "section": "علوم وطبيعة",    "topic": "كيف يعمل الدماغ البشري بشكل مبسط"},
    {"id": 87, "section": "علوم وطبيعة",    "topic": "التغيرات المناخية وتأثيرها على حياتنا"},
    {"id": 88, "section": "علوم وطبيعة",    "topic": "الطاقة الشمسية والطاقة المتجددة"},
    {"id": 89, "section": "علوم وطبيعة",    "topic": "كيف تعمل الطائرة والجاذبية"},
    {"id": 90, "section": "علوم وطبيعة",    "topic": "دورة الماء والتوازن البيئي"},

    {"id": 91, "section": "حياة يومية",     "topic": "ترتيب وتنظيم المنزل بكفاءة"},
    {"id": 92, "section": "حياة يومية",     "topic": "نصائح الطبخ والوصفات للمبتدئين"},
    {"id": 93, "section": "حياة يومية",     "topic": "توفير الكهرباء والماء في المنزل"},
    {"id": 94, "section": "حياة يومية",     "topic": "التخطيط للسفر والرحلات"},
    {"id": 95, "section": "حياة يومية",     "topic": "تربية الأطفال ومراحل النمو"},
    {"id": 96, "section": "حياة يومية",     "topic": "العناية بالمظهر الشخصي"},
    {"id": 97, "section": "حياة يومية",     "topic": "الهوايات وقضاء وقت الفراغ بشكل مفيد"},
    {"id": 98, "section": "حياة يومية",     "topic": "التعامل مع الجيران وآداب التعايش"},
    {"id": 99, "section": "حياة يومية",     "topic": "الاعتناء بالنباتات المنزلية"},
    {"id": 100,"section": "حياة يومية",     "topic": "حل المشكلات اليومية البسيطة باتزان"},
]

# ==============================================================================
#  🔑  مدير مفاتيح API — يتبدل تلقائياً عند الفشل
# ==============================================================================

class APIKeyManager:
    def __init__(self, keys: list[str]):
        valid = [k for k in keys if k and not k.startswith("YOUR_")]
        if not valid:
            raise ValueError("❌ لا توجد مفاتيح API صحيحة. عدّل GEMINI_API_KEYS أولاً.")
        self.keys   = valid
        self.index  = 0
        self.errors = {k: 0 for k in valid}   # عداد الأخطاء لكل مفتاح
        print(f"✅ تم تحميل {len(self.keys)} مفاتيح API")

    @property
    def current_key(self) -> str:
        return self.keys[self.index]

    def get_client(self) -> genai.Client:
        return genai.Client(api_key=self.current_key)

    def rotate(self, reason: str = "") -> bool:
        """الانتقال للمفتاح التالي. يرجع False لو استنفدنا كل المفاتيح."""
        self.errors[self.current_key] += 1
        next_idx = (self.index + 1) % len(self.keys)
        if next_idx == self.index:
            return False   # مفتاح واحد فقط متاح
        self.index = next_idx
        print(f"  🔄 تبديل للمفتاح #{self.index + 1} — السبب: {reason}")
        return True

    def status(self) -> str:
        lines = [f"  مفتاح #{i+1}: {v} أخطاء" for i, (k, v) in enumerate(self.errors.items())]
        return "\n".join(lines)


# ==============================================================================
#  💾  مدير نقطة الاستئناف (Checkpoint)
# ==============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_path: str, output_path: str):
        self.cp_path  = Path(checkpoint_path)
        self.out_path = Path(output_path)

    def load(self) -> tuple[list[dict], set[str]]:
        """يحمّل الداتا المحفوظة ومجموعة المهام المكتملة."""
        dataset   : list[dict] = []
        completed : set[str]   = set()

        if self.out_path.exists():
            try:
                with open(self.out_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)
                print(f"  📂 استُعيدت {len(dataset)} عينة من '{self.out_path}'")
            except Exception as e:
                print(f"  ⚠️  تعذّر تحميل ملف الإخراج: {e}")

        if self.cp_path.exists():
            try:
                with open(self.cp_path, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                completed = set(cp.get("completed_keys", []))
                print(f"  📌 {len(completed)} مهمة مكتملة في نقطة الاستئناف")
            except Exception as e:
                print(f"  ⚠️  تعذّر تحميل نقطة الاستئناف: {e}")

        return dataset, completed

    def save(self, dataset: list[dict], completed: set[str]):
        """يحفظ الداتا ونقطة الاستئناف."""
        _safe_write(self.out_path, dataset)
        cp = {
            "completed_keys": list(completed),
            "count"         : len(dataset),
            "saved_at"      : datetime.now().isoformat()
        }
        _safe_write(self.cp_path, cp)


def _safe_write(path: Path, data):
    """كتابة آمنة: اكتب لملف مؤقت أولاً ثم استبدل."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ==============================================================================
#  ✂️  التحقق من حد الكلمات وتقليم النص
# ==============================================================================

def count_words(text: str) -> int:
    return len(text.split())

def trim_to_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    # أضف نقطة في النهاية لو مش موجودة
    if trimmed and trimmed[-1] not in ".!?،؟":
        trimmed += "."
    return trimmed

def validate_and_fix(example: dict, max_words: int = MAX_WORDS) -> dict:
    """
    يتحقق من بنية المثال ويوزّع الكلمات بشكل متوازن للنموذج 600M:
      system  :  ~30  كلمة  (3%)
      query   :  ~70  كلمة  (8%)
      thought :  ~300 كلمة  (33%)
      answer  :  ~500 كلمة  (55%)
    """
    budgets = {
        "system" : 40,
        "query"  : 80,
        "thought": 320,
        "answer" : 480,
    }
    for field, budget in budgets.items():
        if field in example and isinstance(example[field], str):
            example[field] = trim_to_limit(example[field].strip(), budget)

    # تحقق من وجود المفتاح "باذن الله" في thought
    thought = example.get("thought", "")
    if thought and not thought.startswith("باذن الله"):
        example["thought"] = "باذن الله ساقوم بالتفكير بعمق. " + thought

    return example


# ==============================================================================
#  🤖  بناء البرومبت المُحسَّن لنموذج 600M
# ==============================================================================

PROMPT_TEMPLATE = """\
أنت مساعد لتوليد بيانات تدريب لنموذج لغوي عربي بحجم 600 مليون معامل.

المطلوب: توليد مثال تدريبي واحد فقط للموضوع التالي:
الموضوع: "{topic}"
القسم: "{section}"
رقم العينة: {sample_num} من {total_samples}

⚠️ قيود الحجم الصارمة (لأن النموذج صغير):
- system  : جملة واحدة مختصرة ≤ 35 كلمة
- query   : سؤال واضح ومباشر ≤ 70 كلمة
- thought : تحليل متدرج ومركّز ≤ 300 كلمة (يبدأ حرفاً بـ "باذن الله ساقوم بالتفكير بعمق")
- answer  : إجابة عملية نافعة ≤ 450 كلمة (واضحة، بدون تكرار)

⚠️ تنويع العينات: هذه العينة رقم {sample_num}، لذا اجعل السؤال والإجابة مختلفَين تماماً عن العينات السابقة لنفس الموضوع.

أعد JSON object واحد فقط بدون أي markdown أو مقدمات:
{{"system": "...", "query": "...", "thought": "...", "answer": "..."}}
"""

# ==============================================================================
#  ⚡  دالة توليد عينة واحدة مع إعادة المحاولة
# ==============================================================================

def generate_single_sample(
    topic_entry : dict,
    sample_num  : int,
    key_manager : APIKeyManager,
    max_retries : int = 3
) -> dict | None:

    prompt = PROMPT_TEMPLATE.format(
        topic         = topic_entry["topic"],
        section       = topic_entry["section"],
        sample_num    = sample_num,
        total_samples = SAMPLES_PER_TOPIC
    )

    for attempt in range(1, max_retries + 1):
        try:
            client   = key_manager.get_client()
            response = client.models.generate_content(
                model    = MODEL_ID,
                contents = prompt
            )

            raw = response.text.strip()

            # تنظيف markdown
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            # استخرج أول كائن JSON
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                raise ValueError("لم يُعثر على JSON في الرد")
            example = json.loads(match.group())

            # تحقق من الحقول الأساسية
            for field in ("system", "query", "thought", "answer"):
                if field not in example or not isinstance(example[field], str):
                    raise ValueError(f"حقل مفقود أو خاطئ: {field}")

            # تطبيق حدود الكلمات والتوازن
            example = validate_and_fix(example)

            # إضافة metadata
            example["_topic_id"] = topic_entry["id"]
            example["_topic"]    = topic_entry["topic"]
            example["_section"]  = topic_entry["section"]
            example["_sample"]   = sample_num

            return example

        except Exception as e:
            err_msg = str(e)
            print(f"    ⚠️  محاولة {attempt}/{max_retries} فشلت: {err_msg[:80]}")

            # تبديل المفتاح عند أخطاء معينة
            rate_err = any(x in err_msg.lower() for x in
                           ["quota", "rate", "429", "limit", "key", "api"])
            if rate_err:
                key_manager.rotate(err_msg[:60])
                time.sleep(3)
            elif attempt < max_retries:
                time.sleep(2 * attempt)

    return None  # فشل بعد كل المحاولات


# ==============================================================================
#  🚀  المحرك الرئيسي
# ==============================================================================

def run_generation():
    print("=" * 60)
    print("  مولّد داتا التدريب — 1000 عينة | نموذج 600M")
    print("=" * 60)

    # تهيئة
    key_manager = APIKeyManager(GEMINI_API_KEYS)
    cp_manager  = CheckpointManager(CHECKPOINT_FILE, OUTPUT_FILE)

    # تحميل ما سبق إنجازه
    dataset, completed = cp_manager.load()
    failed_tasks: list[dict] = []

    # بناء قائمة المهام الكاملة: كل موضوع × SAMPLES_PER_TOPIC
    all_tasks = [
        {"topic_entry": t, "sample_num": s}
        for t in TOPICS
        for s in range(1, SAMPLES_PER_TOPIC + 1)
    ]
    total_tasks = len(all_tasks)

    # تصفية المهام المكتملة
    def task_key(t: dict) -> str:
        return f"{t['topic_entry']['id']}_{t['sample_num']}"

    pending = [t for t in all_tasks if task_key(t) not in completed]

    print(f"\n📊 الإجماليات:")
    print(f"   المهام الكلية  : {total_tasks}")
    print(f"   مكتمل          : {total_tasks - len(pending)}")
    print(f"   متبقي للتوليد  : {len(pending)}")
    print(f"   حد الكلمات     : {MAX_WORDS} كلمة/عينة\n")

    if not pending:
        print("✅ كل المهام مكتملة بالفعل!")
        _print_summary(dataset)
        return dataset

    start_time = time.time()

    try:
        for i, task in enumerate(pending, start=1):
            te  = task["topic_entry"]
            sn  = task["sample_num"]
            key = task_key(task)

            print(f"[{i:>4}/{len(pending)}] "
                  f"§{te['section']:<16} | "
                  f"#{te['id']:>3} {te['topic'][:35]:<35} | "
                  f"عينة {sn}/{SAMPLES_PER_TOPIC}")

            result = generate_single_sample(te, sn, key_manager)

            if result:
                dataset.append(result)
                completed.add(key)
                total_words = sum(
                    count_words(result.get(f, ""))
                    for f in ("system", "query", "thought", "answer")
                )
                print(f"         ✅ تم | {total_words} كلمة")
            else:
                failed_tasks.append(task)
                print(f"         ❌ فشل نهائي — سيُحفظ للمراجعة")

            # حفظ دوري
            if i % SAVE_EVERY == 0:
                cp_manager.save(dataset, completed)
                elapsed = time.time() - start_time
                rate    = i / elapsed * 60
                eta_min = (len(pending) - i) / rate if rate > 0 else 0
                print(f"\n  💾 حفظ تلقائي — {len(dataset)} عينة | "
                      f"الوتيرة: {rate:.1f}/دقيقة | "
                      f"المتبقي: ~{eta_min:.0f} دقيقة\n")

            time.sleep(DELAY_BETWEEN_CALLS)

    except KeyboardInterrupt:
        print("\n\n⛔ توقف يدوي — جاري الحفظ الطارئ...")

    finally:
        # حفظ نهائي دائماً
        cp_manager.save(dataset, completed)

        # حفظ المهام الفاشلة
        if failed_tasks:
            _safe_write(Path(FAILED_FILE), failed_tasks)
            print(f"\n⚠️  {len(failed_tasks)} مهمة فاشلة حُفظت في '{FAILED_FILE}'")

        print(f"\n{key_manager.status()}")

    _print_summary(dataset)
    return dataset


# ==============================================================================
#  🔁  إعادة معالجة المهام الفاشلة (تشغيل منفصل)
# ==============================================================================

def retry_failed():
    if not Path(FAILED_FILE).exists():
        print("لا يوجد ملف مهام فاشلة.")
        return

    with open(FAILED_FILE, "r", encoding="utf-8") as f:
        failed_tasks = json.load(f)

    if not failed_tasks:
        print("قائمة المهام الفاشلة فارغة.")
        return

    print(f"🔄 إعادة معالجة {len(failed_tasks)} مهمة فاشلة...")

    key_manager = APIKeyManager(GEMINI_API_KEYS)
    cp_manager  = CheckpointManager(CHECKPOINT_FILE, OUTPUT_FILE)
    dataset, completed = cp_manager.load()

    still_failed = []
    for task in failed_tasks:
        te  = task["topic_entry"]
        sn  = task["sample_num"]
        key = f"{te['id']}_{sn}"

        print(f"  ↺ {te['topic']} — عينة {sn}")
        result = generate_single_sample(te, sn, key_manager)

        if result:
            dataset.append(result)
            completed.add(key)
            print(f"    ✅ نجح")
        else:
            still_failed.append(task)
            print(f"    ❌ فشل مجدداً")

        time.sleep(DELAY_BETWEEN_CALLS)

    cp_manager.save(dataset, completed)

    if still_failed:
        _safe_write(Path(FAILED_FILE), still_failed)
        print(f"\n{len(still_failed)} مهمة لا تزال فاشلة.")
    else:
        Path(FAILED_FILE).unlink(missing_ok=True)
        print("\n✅ تمت معالجة جميع المهام الفاشلة!")

    _print_summary(dataset)


# ==============================================================================
#  📊  ملخص إحصائي
# ==============================================================================

def _print_summary(dataset: list[dict]):
    if not dataset:
        print("\n(لا توجد عينات)")
        return

    # إحصاء الأقسام
    sections: dict[str, int] = {}
    word_counts = []
    for ex in dataset:
        sec = ex.get("_section", "غير محدد")
        sections[sec] = sections.get(sec, 0) + 1
        total_w = sum(count_words(ex.get(f, "")) for f in ("system","query","thought","answer"))
        word_counts.append(total_w)

    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    max_words_found = max(word_counts) if word_counts else 0
    over_limit = sum(1 for w in word_counts if w > MAX_WORDS)

    print(f"\n{'='*60}")
    print(f"  ✅ بفضل الله — ملخص التوليد")
    print(f"{'='*60}")
    print(f"  إجمالي العينات : {len(dataset)}")
    print(f"  متوسط الكلمات  : {avg_words:.0f}")
    print(f"  أعلى عدد كلمات : {max_words_found}")
    print(f"  تجاوز الحد     : {over_limit} عينة")
    print(f"\n  توزيع الأقسام:")
    for sec, cnt in sorted(sections.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 2)
        print(f"    {sec:<18} {cnt:>4}  {bar}")
    print(f"\n  📁 الملف: '{OUTPUT_FILE}'")
    print(f"{'='*60}")


# ==============================================================================
#  🎬  نقطة الدخول
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--retry-failed":
        retry_failed()
    else:
        run_generation()
