import threading


class ChatMemory:
    def __init__(self):
        self.lock = threading.Lock()
        self._store = {}

    def add_message(self, session_id: str, user_msg: str, bot_msg: str):
        with self.lock:
            if session_id not in self._store:
                self._store[session_id] = []
                
            self._store[session_id].append({
                "user": user_msg,
                "bot": bot_msg
            })

    def get_messages(self, session_id: str):
        with self.lock:
            return list(self._store.get(session_id, []))
    def get_all_messages(self):
        with self.lock:
            return dict(self._store)
          
    def clear_session(self, session_id: str):
        with self.lock:
            self._store.pop(session_id, None)