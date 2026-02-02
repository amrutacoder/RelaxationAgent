"""Logging and database module."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from .config import Config


class Logger:
    """Handles logging to database and file."""
    
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.log_file = Config.LOG_FILE
        self._setup_file_logging()
        self._setup_database()
    
    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("relaxation_agent")
    
    def _setup_database(self):
        """Setup SQLite database for persistent logging."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                text_input TEXT,
                emotion_probs TEXT,
                top_emotion TEXT,
                stress_score REAL,
                stress_level TEXT,
                prompt TEXT,
                audio_features TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stress_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                stress_score REAL,
                stress_level TEXT,
                emotion TEXT,
                notified_agents TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def log_emotion_analysis(
        self,
        text_input: Optional[str],
        emotion_probs: Dict[str, float],
        top_emotion: str,
        stress_score: float,
        stress_level: str,
        prompt: str,
        user_id: Optional[str] = None,
        audio_features: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Log emotion analysis result to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO emotion_logs (
                timestamp, user_id, text_input, emotion_probs, top_emotion,
                stress_score, stress_level, prompt, audio_features, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            user_id,
            text_input,
            json.dumps(emotion_probs),
            top_emotion,
            stress_score,
            stress_level,
            prompt,
            json.dumps(audio_features) if audio_features else None,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(
            f"Logged emotion analysis: {top_emotion} (stress: {stress_score:.2f}, level: {stress_level})"
        )
    
    def log_stress_alert(
        self,
        stress_score: float,
        stress_level: str,
        emotion: str,
        user_id: Optional[str] = None,
        notified_agents: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """Log stress alert to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO stress_alerts (
                timestamp, user_id, stress_score, stress_level, emotion,
                notified_agents, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            user_id,
            stress_score,
            stress_level,
            emotion,
            json.dumps(notified_agents) if notified_agents else None,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.warning(
            f"Logged stress alert: {stress_level} stress ({stress_score:.2f}) - {emotion}"
        )
    
    def get_emotion_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Retrieve emotion analysis history."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT * FROM emotion_logs
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM emotion_logs
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_stress_alerts(
        self,
        user_id: Optional[str] = None,
        stress_level: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Retrieve stress alert history."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM stress_alerts WHERE 1=1'
        params = []
        
        if user_id:
            query += ' AND user_id = ?'
            params.append(user_id)
        
        if stress_level:
            query += ' AND stress_level = ?'
            params.append(stress_level)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

