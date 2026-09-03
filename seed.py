"""
Seed the prototype with users, groups, on-topic chat history, and per-group
documents, then build each group's vector index.

Idempotent: if any users already exist it does nothing, so it's safe to call
on every server startup. Delete app.db (and the kb_documents/*/ seed files if
you want a totally clean slate) to reseed from scratch.

All passwords are "password" for local testing. Change hash_password inputs
if you care.
"""
import time
import uuid
from pathlib import Path

import db
import auth
import rag

KB_ROOT = Path(__file__).parent / "kb_documents"
ATTACH_DIR = Path(__file__).parent / "attachments"   # mirrors server.py's ATTACH_DIR

# --------------------------------------------------------------------------
# users
# --------------------------------------------------------------------------
USERS = [
    # id            username        display name            role               avatar
    ("dev1",        "mangesh",     "Mangesh",              "developer",        "🧑‍💻"),
    ("pm1",         "rutvik",      "Rutvik Surve",         "project_manager",  "🧭"),
    ("dev2",        "sanjana",     "Sanjana Deshmukh",     "developer",        "👩‍💻"),
    ("dev3",        "aniket",      "Aniket Gade",          "developer",        "🧑‍💻"),
    ("qa1",         "tafazzul",    "Tafazzul Ansari",      "qa",               "🧪"),
    ("dev4",        "shivalila",   "Shivalila Patil",      "developer",        "👩‍💻"),
    ("dev5",        "mrunal",      "Mrunal Khanvilkar",    "developer",        "👨‍💻"),
    ("stakeholder1","saurabh",     "Saurabh",              "stakeholder",      "🕴️"),
]

# --------------------------------------------------------------------------
# groups + membership
# --------------------------------------------------------------------------
GROUPS = [
    ("grp_doctor_opd_internal", "Doctor OPD Internal Group",
     ["dev1", "pm1", "dev2", "dev3", "qa1", "dev4", "dev5", "stakeholder1"]),
]

# --------------------------------------------------------------------------
# chat history for the group. (sender_id, text). Roles/names looked up from
# USERS. 60 messages: the original 50-message dev thread (deployment updates,
# bug triage, WFH/leave scheduling, build/release sharing, sprint check-ins)
# plus 10 messages from Saurabh (the group's only stakeholder), interspersed
# throughout so his questions land near the topic they're about. Most of his
# lines end in '?', which flags is_question=True and drives the PM-suggestion
# panel (server.py only routes stakeholder QUESTIONs to project_manager members).
# --------------------------------------------------------------------------
HISTORY = {
    "grp_doctor_opd_internal": [
        ('dev1', "I'll be a bit late, we will connect once I reach office."),
        ('dev1', 'Sanjana Deshmukh, can we connect?'),
        ('dev2', "Let's connect quickly."),
        ('dev1', 'Hi Aniket Gade, for the issue "Appointment Booked by Receptionist Shows Blank Card in Calendar", could you please share the endpoint?\nCC: Sanjana Deshmukh'),
        ('dev1', 'Deployment is in progress on internal server'),
        ('pm1', 'Leave / WFH Schedule:\n1) 28 August - WFH (Raksha Bandhan)\n2) 25 Sept - WFH (due to Anant Chaturthi - traffic issue)\n3) 26 & 27 November - Leave\nCC: Rutvik Surve'),
        ('stakeholder1', 'Just flagging: our clinic ops team asked about the WFH schedule for late September — wanted to confirm the 25th is correct.'),
        ('dev1', 'Deployment Done'),
        ('dev1', 'Aniket Gade Sanjana Deshmukh Rutvik Surve\nDeployment Update:\n\nDoctorService.GetAllDoctor() – Pagination Fix\nFixed the issue where PageSize = 10 returned fewer records because pagination was applied before establishment/location filtering. Updated the filtering and pagination order, and updated TotalCount and NextPageNumber calculations. Verified and confirmed the fix is working as expected.\n\nFix appointment history retrieval for receptionist users\nUpdated AppointmentService.GetPatientAppointmentHistory() and AppointmentRepository.GetPatientAppointmentHistory(). Changed appointment lookup to use PatientId and AppointmentId instead of requiring the logged-in user to have a Doctor profile — DoctorId is now derived from the retrieved appointment. This allows both doctors and receptionists to fetch appointment history successfully.'),
        ('stakeholder1', 'Good to hear about the pagination fix — how many establishments were affected before it was patched?'),
        ('dev1', 'Tafazzul Ansari, please use the attached latest backend build for the PreProd env.\nDoctorOPDBuild_26_08_2026.zip'),
        ('dev1', 'Deployment is in progress on internal server'),
        ('dev1', 'Deployment Done'),
        ('dev1', 'Shivalila Patil Sanjana Deshmukh Aniket Gade\nDeployment Update:\nFixed DoctorService.GetDoctor() – BookingDisable Logic. Updated the GetDoctor() method to apply the same 14-day booking window logic used in GetAllDoctor(). Booking is now disabled when no slot is available or when the next available slot falls after the 14-day window. Verified and working as expected.'),
        ('dev1', 'CC: Rutvik Surve\nThe Excel sheet has been updated with the deployed changes.'),
        ('dev1', 'Deployment is in progress on internal server'),
        ('dev1', 'Deployment Done\nShivalila Patil Sanjana Deshmukh Aniket Gade\n\nDeployment Update:\nFix Patient Details Retrieval for Duplicate Mobile Numbers – Updated PatientRepository.Get(string mobileNo) to filter by RoleId=3 (Patient) along with the mobile number. This ensures that when the same mobile number is registered for both a Doctor and a Patient, the phone/get API returns only the Patient details. Verified and working as expected.'),
        ('stakeholder1', 'Appreciate the team pushing through the duplicate mobile number fix quickly, that was blocking onboarding for a partner clinic.'),
        ('dev1', 'CC: Rutvik Surve\nThe Excel sheet has been updated with the deployed changes.'),
        ('dev1', 'Shivalila Patil, please let me know the tasks you have covered this week. For sprint planning I need the updates accordingly. You can also edit in the sheet you shared and mark as Done.\nCC: Rutvik Surve'),
        ('dev4', 'I have completed adding the Frontend tasks for the Admin Portal and Teleconsultation epics.'),
        ('dev1', 'Mrunal Khanvilkar, please let me know the tasks you have covered this week. For sprint planning I need the updates accordingly. You can also edit in the sheet you shared and mark as Done.\nCC: Rutvik Surve'),
        ('dev5', 'I have completed adding the Backend tasks for the Admin Portal and Teleconsultation epics.'),
        ('dev1', 'Important!\nRutvik Surve Sanjana Deshmukh\nI have shared new Release APK on mail. Please check. Thanks!'),
        ('dev1', "Shivalila Patil Mrunal Khanvilkar\nPlease update Azure on priority: move the last week's tasks assigned to you to Dev Completed once they are completed, and add the time taken/required to complete each task in the Completed field.\ncc: Rutvik Surve"),
        ('dev3', 'Created a new doctor from mobile — getting a 500 Internal Server Error. Also, a pop-up is displayed on the Doctor side.'),
        ('dev3', 'https://staging-api.doctoropd.com/establishment/get\n{"Data":"kldStb200PKtegYkfKVMWERIxt2WDx9+VMTpMG3+qaTKjpOYJqGdzS+/7N/tDZ/N1bd+YPFqvgHOYrOFX5GG1whR1+9VjAfmL3VjH6Oo9uU5Ea2U0tIFAiPH+KwC6S3O"}\nResponse looks encrypted/garbled — is that expected on this endpoint?'),
        ('stakeholder1', 'Why was the establishment/get response coming back encrypted — is that intentional or a bug?'),
        ('dev1', 'We have tested the Pre-Prod build and found a few issues. The issues have been fixed and re-tested successfully on debug apk. We are now proceeding with the Production APK release.\nCC: Rutvik Surve'),
        ('dev1', 'Important!\nSanjana Deshmukh Rutvik Surve\nI have shared Production APK on mail for Testing. Thanks!'),
        ('stakeholder1', 'Is the Production APK release confirmed live for all establishments now?'),
        ('dev2', "Concern related to these doctors, let's discuss in the standup.\nScreenshot 2026-09-02 at 11.47.23 AM.png"),
        ('dev1', 'Mrunal Khanvilkar, whatever changes you done in preprod, are they all deployed to Production?\ncc Sanjana Deshmukh Rutvik Surve'),
        ('dev1', 'Deployment is in progress on internal server'),
        ('dev5', 'Deployment Done. Whatever was in PreProd is now live on Production.'),
        ('dev1', "Thanks Mrunal. Closing out this week's deployment tracker."),
        ('pm1', "Standup notes — carrying over the doctor 500 error and the encrypted establishment/get response as today's top priority."),
        ('dev3', 'Debugged the establishment/get issue — the response is AES-encrypted by design on that endpoint, we were missing the decrypt step on the new-doctor-from-mobile flow. Fix in progress.'),
        ('dev1', "Good catch. Once that's patched, let's also verify the same decrypt step is applied consistently across all establishment-scoped endpoints, not just this one."),
        ('dev1', 'Deployment Update: Added SARVAM as a third STT provider option (STTS=SARVAM) alongside the existing DEEPMURF/OPENAI switch in voice_provider.py, for evaluation on the voice booking flow.'),
        ('stakeholder1', 'Does the new Sarvam STT option handle Kannada as well, or just Hindi so far?'),
        ('dev2', "Ran a quick comparison call on the IVR side — Sarvam's Hindi transcription looked noticeably cleaner on digit-heavy utterances (mobile numbers, OTP)."),
        ('dev1', 'Deployment Update: Fixed specialist ranking to use the full QnA context (chief complaint + clarifying answers + allergy + medications) instead of just the opening message. Should improve specialization accuracy for ambiguous symptoms.'),
        ('stakeholder1', 'Can someone confirm whether the specialist ranking fix impacts response time for patients during booking?'),
        ('pm1', "Nice, that's the ranking issue from Tuesday's review. Any impact on cache hit rate?"),
        ('dev1', "Some — the cache keys are now more specific so we get a few more misses, but we're prioritizing accuracy over cache hit rate here. Logged as an accepted trade-off."),
        ('dev4', 'Frontend heads up: pushed the multilingual card UI changes — doctor/establishment names now transliterate into the selected language script for display, English values still sent to backend on selection.'),
        ('dev1', 'Confirmed on staging, looks correct in Hindi and Marathi. Kannada rendering needs a font check — raising a separate ticket.'),
        ('stakeholder1', 'When can we expect the Kannada rendering fix to be verified and closed out?'),
        ('qa1', 'Regression pass on the PreProd build complete — appointment booking, cancellation, and reschedule flows all pass across English/Hindi/Marathi. Kannada font issue noted, otherwise clean.'),
        ('dev1', 'Thanks Tafazzul Ansari. Attaching the current sprint task tracker for reference — Admin Portal and Teleconsultation status for everyone.'),
        ('stakeholder1', "Thanks for sharing the sprint task tracker, it's helpful for our steering review."),
        ('dev5', 'Got it, updating my Azure items against this.'),
        ('dev1', 'Deployment is in progress on internal server'),
        ('dev1', 'Deployment Done — FAISS index audit fixes for the doctors, specializations, and establishments indexes are live. Also fixed the ConsultationFee/consultation_fee key mismatch between FAISS and the DB fallback path.'),
        ('stakeholder1', 'Is the FAISS index audit fix already live on Production, or still only on PreProd?'),
        ('dev3', 'Verified — establishment search now falls back to DB correctly when FAISS misses. No more silent empty results.'),
        ('pm1', "Good progress this week. Sanjana Deshmukh, can you compile the endpoint reference doc we discussed so new QA folks aren't pinging Mangesh directly for URLs?"),
        ('dev2', 'On it — putting together the API endpoint reference now, will drop it in the group docs.'),
        ('dev1', "Added — api_endpoints_reference.md is up in the group's knowledge base folder. Thanks Sanjana Deshmukh."),
    ],
}


# index into HISTORY[gid] (0-based) of the message that the in-chat
# attachment rides in on
ATTACHMENT_MESSAGE_INDEX = {
    "grp_doctor_opd_internal": 50,   # "...Attaching the current sprint task tracker..."
}

# --------------------------------------------------------------------------
# per-group seed documents, written to ./kb_documents/<group_id>/
# (root-folder docs, dropped outside the chat UI — RAG source 'doc')
# --------------------------------------------------------------------------
DOCS = {
    "grp_doctor_opd_internal": {
        "api_endpoints_reference.md": """# DoctorOPD Backend - API Endpoint Reference

## Establishment
- GET  /establishment/get
  Returns establishment data for the logged-in context. Response payload is
  AES-encrypted by design (`{"Data": "<ciphertext>"}`); callers must run the
  standard decrypt step before use. The new-doctor-from-mobile flow was
  missing this step, which caused the 500 + doctor-side pop-up bug — fix is
  to apply the same decrypt helper used elsewhere for establishment-scoped
  endpoints.

## Doctor
- GET  /doctor/getAllDoctor (DoctorService.GetAllDoctor())
  Paginated doctor listing. Filtering (establishment/location) is applied
  BEFORE pagination, not after — PageSize=10 will silently return fewer than
  10 records if filtering happens post-pagination. TotalCount and
  NextPageNumber are derived from the post-filter set.
- GET  /doctor/get (DoctorService.GetDoctor())
  Single-doctor lookup. BookingDisable is computed with the same 14-day
  booking-window rule as GetAllDoctor(): disabled when no slot is available,
  or when the next available slot falls outside the 14-day window.

## Appointment
- GET  /appointment/history (AppointmentService.GetPatientAppointmentHistory
  / AppointmentRepository.GetPatientAppointmentHistory)
  Looks up appointment history by PatientId + AppointmentId — does NOT
  require the logged-in user to have a Doctor profile. DoctorId is derived
  from the retrieved appointment, so both doctor and receptionist logins can
  fetch history successfully.

## Patient
- GET  /phone/get (PatientRepository.Get(string mobileNo))
  Looks up a patient by mobile number, filtered to RoleId=3 (Patient). This
  filter exists specifically to avoid returning Doctor details when the same
  mobile number is registered against both a Doctor and a Patient account.

## Notes for QA / new joiners
- All establishment-scoped GET endpoints return encrypted payloads — if a
  response looks like base64 garbage instead of JSON, that's expected; check
  the decrypt step before filing it as a bug.
- Staging base URL: https://staging-api.doctoropd.com
""",
    },
}

# --------------------------------------------------------------------------
# per-group in-chat attachments: (filename, mime, content) rides in on the
# message at ATTACHMENT_MESSAGE_INDEX[gid]. RAG source 'attachment'.
# --------------------------------------------------------------------------
ATTACHMENTS = {
    "grp_doctor_opd_internal": [
        ("sprint_task_tracker.md", "text/markdown", """# Sprint Task Tracker - Admin Portal & Teleconsultation

## Admin Portal
| Task                                   | Owner              | Status        | Time (hrs) |
|-----------------------------------------|--------------------|--------------|-----------|
| Establishment list + filters (UI)       | Shivalila Patil    | Done          | 6         |
| Doctor pagination fix verification      | Mrunal Khanvilkar  | Done          | 3         |
| BookingDisable 14-day window (backend)  | Mrunal Khanvilkar  | Done          | 5         |
| Admin doctor-create pop-up bug          | Aniket Gade        | In Progress   | -         |

## Teleconsultation
| Task                                   | Owner              | Status        | Time (hrs) |
|-----------------------------------------|--------------------|--------------|-----------|
| Appointment history for receptionist    | Mrunal Khanvilkar  | Done          | 4         |
| Multilingual card UI (transliteration)  | Shivalila Patil    | Done          | 7         |
| Kannada font rendering fix              | Shivalila Patil    | To Do         | -         |
| Sarvam STT evaluation (voice booking)   | Mangesh            | In Progress   | -         |

## Notes
- Move completed items to "Dev Completed" on Azure and log actual time taken
  in the Completed field per the priority ask.
- Regression pass owner for this sprint: Tafazzul Ansari.
"""),
    ],
}


def _user_lookup():
    return {u[0]: u for u in USERS}


def run():
    db.init_db()

    # Idempotency: check each user/group individually (by id), not a single
    # hardcoded username. This makes seeding safe to re-run even if a prior
    # attempt crashed or was killed partway through (e.g. only some users
    # got inserted before the process died) — we just fill in what's missing
    # instead of blindly re-inserting everything and hitting UNIQUE errors.
    print("[seed] checking users…")
    for uid, username, name, role, avatar in USERS:
        if db.get_user(uid):
            continue
        db.create_user(uid, username, name, role, avatar, auth.hash_password("password"))

    umap = _user_lookup()

    print("[seed] creating groups, membership, history, attachments, and docs…")
    for gid, gname, members in GROUPS:
        is_new_group = not db.group_exists(gid)
        if is_new_group:
            db.create_group(gid, gname)
        for m in members:
            db.add_member(gid, m)   # INSERT OR IGNORE under the hood, safe to repeat

        if not is_new_group:
            continue  # history/docs/attachments already written for this group on a prior run

        # chat history (spaced timestamps so ordering is stable)
        base = time.time() - 86400
        att_idx = ATTACHMENT_MESSAGE_INDEX.get(gid)
        att_msg_id = None
        for i, (sender_id, text) in enumerate(HISTORY[gid]):
            _, _, sname, srole, _ = umap[sender_id]
            # stakeholder questions get flagged so the panel logic can use them
            is_q = srole == "stakeholder" and text.strip().endswith("?")
            msg = db.add_message(gid, sender_id, sname, srole, text, is_question=is_q)
            # backdate for deterministic ordering
            with db.get_db() as conn:
                conn.execute("UPDATE messages SET created_at = ? WHERE id = ?", (base + i, msg["id"]))
            if att_idx is not None and i == att_idx:
                att_msg_id = msg["id"]

        # write the in-chat attachment(s) to ./attachments/<gid>/ and register
        # them against the message they rode in on, same layout the live
        # upload route (server.py) uses: <group_id>/<uuid>_<filename>
        for fname, mime, content in ATTACHMENTS.get(gid, []):
            gdir = ATTACH_DIR / gid
            gdir.mkdir(parents=True, exist_ok=True)
            raw = content.encode("utf-8")
            stored_name = f"{uuid.uuid4().hex}_{fname}"
            (gdir / stored_name).write_bytes(raw)
            rel_path = f"{gid}/{stored_name}"
            db.add_attachment(gid, att_msg_id, fname, rel_path, mime, len(raw), content)

        # write seed docs to ./kb_documents/<gid>/
        group_dir = KB_ROOT / gid
        group_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in DOCS[gid].items():
            (group_dir / fname).write_text(content, encoding="utf-8")

    print("[seed] building per-group vector indexes…")
    for gid, _, _ in GROUPS:
        n = rag.build_group_index(gid)
        print(f"[seed]   {gid}: {n} chunks")

    print("[seed] done. Logins: mangesh (developer), rutvik (project_manager), "
          "sanjana / aniket / shivalila / mrunal (developer), tafazzul (qa), "
          "saurabh (stakeholder) — password: 'password'")


if __name__ == "__main__":
    run()
