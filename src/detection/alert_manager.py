"""
Alert management system for intrusion detection
"""
import smtplib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict, deque
import requests
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alerts and notifications for intrusion detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alert_history = deque(maxlen=10000)
        self.alert_rules = {}
        self.notification_channels = {}
        self.alert_callbacks = []
        
        # Alert aggregation settings
        self.aggregation_window = timedelta(minutes=5)
        self.max_alerts_per_window = 10
        
        # Initialize notification channels
        self._initialize_notification_channels()
        
    def _initialize_notification_channels(self):
        """Initialize notification channels"""
        # Email configuration
        if 'email' in self.config:
            self.notification_channels['email'] = {
                'enabled': True,
                'smtp_server': self.config['email'].get('smtp_server', 'smtp.gmail.com'),
                'smtp_port': self.config['email'].get('smtp_port', 587),
                'username': self.config['email'].get('username'),
                'password': self.config['email'].get('password'),
                'recipients': self.config['email'].get('recipients', [])
            }
        
        # Slack configuration
        if 'slack' in self.config:
            self.notification_channels['slack'] = {
                'enabled': True,
                'webhook_url': self.config['slack'].get('webhook_url'),
                'channel': self.config['slack'].get('channel', '#security-alerts')
            }
        
        # Webhook configuration
        if 'webhook' in self.config:
            self.notification_channels['webhook'] = {
                'enabled': True,
                'url': self.config['webhook'].get('url'),
                'headers': self.config['webhook'].get('headers', {})
            }
    
    def add_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """Add an alert rule"""
        self.alert_rules[rule_name] = {
            'enabled': rule_config.get('enabled', True),
            'conditions': rule_config.get('conditions', {}),
            'severity': rule_config.get('severity', 'medium'),
            'channels': rule_config.get('channels', ['email']),
            'cooldown': timedelta(minutes=rule_config.get('cooldown_minutes', 15)),
            'last_triggered': None
        }
        logger.info(f"Added alert rule: {rule_name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add custom alert callback"""
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")
    
    def process_alert(self, alert_data: Dict[str, Any]):
        """Process an incoming alert"""
        alert_record = {
            'id': self._generate_alert_id(),
            'timestamp': datetime.now(),
            'alert_level': alert_data.get('alert_level', 'medium'),
            'ensemble_prediction': alert_data.get('ensemble_prediction', 0),
            'confidence_scores': alert_data.get('confidence_scores', {}),
            'individual_predictions': alert_data.get('individual_predictions', {}),
            'processed': False,
            'notifications_sent': []
        }
        
        # Check if alert should be processed
        if self._should_process_alert(alert_record):
            # Apply alert rules
            triggered_rules = self._check_alert_rules(alert_record)
            
            if triggered_rules:
                # Send notifications
                self._send_notifications(alert_record, triggered_rules)
                alert_record['processed'] = True
                alert_record['triggered_rules'] = triggered_rules
            
            # Call custom callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_record)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
        
        # Store alert record
        self.alert_history.append(alert_record)
        
        return alert_record
    
    def _should_process_alert(self, alert_record: Dict[str, Any]) -> bool:
        """Check if alert should be processed based on aggregation rules"""
        current_time = alert_record['timestamp']
        
        # Count recent alerts of the same level
        recent_alerts = [
            alert for alert in self.alert_history
            if (current_time - alert['timestamp']) <= self.aggregation_window
            and alert['alert_level'] == alert_record['alert_level']
        ]
        
        # Don't process if too many alerts in the window
        if len(recent_alerts) >= self.max_alerts_per_window:
            logger.warning(f"Too many {alert_record['alert_level']} alerts in window, skipping")
            return False
        
        return True
    
    def _check_alert_rules(self, alert_record: Dict[str, Any]) -> List[str]:
        """Check which alert rules are triggered"""
        triggered_rules = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue
            
            # Check cooldown
            if rule['last_triggered']:
                time_since_last = alert_record['timestamp'] - rule['last_triggered']
                if time_since_last < rule['cooldown']:
                    continue
            
            # Check conditions
            if self._evaluate_rule_conditions(rule['conditions'], alert_record):
                triggered_rules.append(rule_name)
                rule['last_triggered'] = alert_record['timestamp']
        
        return triggered_rules
    
    def _evaluate_rule_conditions(self, conditions: Dict[str, Any], alert_record: Dict[str, Any]) -> bool:
        """Evaluate alert rule conditions"""
        for condition, value in conditions.items():
            if condition == 'min_alert_level':
                alert_levels = {'normal': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                if alert_levels.get(alert_record['alert_level'], 0) < alert_levels.get(value, 0):
                    return False
            
            elif condition == 'min_confidence':
                if alert_record['confidence_scores']:
                    max_confidence = max(alert_record['confidence_scores'].values())
                    if max_confidence < value:
                        return False
            
            elif condition == 'ensemble_prediction':
                if alert_record['ensemble_prediction'] != value:
                    return False
            
            elif condition == 'model_agreement':
                predictions = list(alert_record['individual_predictions'].values())
                if len(predictions) > 0:
                    agreement = sum(predictions) / len(predictions)
                    if agreement < value:
                        return False
        
        return True
    
    def _send_notifications(self, alert_record: Dict[str, Any], triggered_rules: List[str]):
        """Send notifications through configured channels"""
        for rule_name in triggered_rules:
            rule = self.alert_rules[rule_name]
            channels = rule['channels']
            
            for channel in channels:
                if channel in self.notification_channels:
                    try:
                        if channel == 'email':
                            self._send_email_notification(alert_record, rule_name)
                        elif channel == 'slack':
                            self._send_slack_notification(alert_record, rule_name)
                        elif channel == 'webhook':
                            self._send_webhook_notification(alert_record, rule_name)
                        
                        alert_record['notifications_sent'].append({
                            'channel': channel,
                            'rule': rule_name,
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error sending {channel} notification: {str(e)}")
    
    def _send_email_notification(self, alert_record: Dict[str, Any], rule_name: str):
        """Send email notification"""
        email_config = self.notification_channels['email']
        
        if not email_config['enabled'] or not email_config['recipients']:
            return
        
        # Create email content
        subject = f"Security Alert - {alert_record['alert_level'].upper()}"
        body = self._create_alert_message(alert_record, rule_name)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['username']
        msg['To'] = ', '.join(email_config['recipients'])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['username'], email_config['password'])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email notification sent for rule: {rule_name}")
    
    def _send_slack_notification(self, alert_record: Dict[str, Any], rule_name: str):
        """Send Slack notification"""
        slack_config = self.notification_channels['slack']
        
        if not slack_config['enabled'] or not slack_config['webhook_url']:
            return
        
        # Create Slack message
        message = {
            "channel": slack_config['channel'],
            "username": "Security Bot",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": self._get_alert_color(alert_record['alert_level']),
                    "title": f"Security Alert - {alert_record['alert_level'].upper()}",
                    "text": self._create_alert_message(alert_record, rule_name),
                    "footer": "AI Intrusion Detection System",
                    "ts": int(alert_record['timestamp'].timestamp())
                }
            ]
        }
        
        # Send to Slack
        response = requests.post(slack_config['webhook_url'], json=message)
        response.raise_for_status()
        
        logger.info(f"Slack notification sent for rule: {rule_name}")
    
    def _send_webhook_notification(self, alert_record: Dict[str, Any], rule_name: str):
        """Send webhook notification"""
        webhook_config = self.notification_channels['webhook']
        
        if not webhook_config['enabled'] or not webhook_config['url']:
            return
        
        # Create webhook payload
        payload = {
            'alert_id': alert_record['id'],
            'timestamp': alert_record['timestamp'].isoformat(),
            'alert_level': alert_record['alert_level'],
            'rule_name': rule_name,
            'ensemble_prediction': alert_record['ensemble_prediction'],
            'confidence_scores': alert_record['confidence_scores'],
            'individual_predictions': alert_record['individual_predictions']
        }
        
        # Send webhook
        response = requests.post(
            webhook_config['url'],
            json=payload,
            headers=webhook_config['headers']
        )
        response.raise_for_status()
        
        logger.info(f"Webhook notification sent for rule: {rule_name}")
    
    def _create_alert_message(self, alert_record: Dict[str, Any], rule_name: str) -> str:
        """Create alert message content"""
        message = f"""
Security Alert Detected

Alert ID: {alert_record['id']}
Timestamp: {alert_record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Alert Level: {alert_record['alert_level'].upper()}
Rule Triggered: {rule_name}

Detection Details:
- Ensemble Prediction: {alert_record['ensemble_prediction']}
- Confidence Scores: {alert_record['confidence_scores']}
- Individual Predictions: {alert_record['individual_predictions']}

This alert was generated by the AI-powered Intrusion Detection System.
Please investigate this potential security incident immediately.

---
AI Cybersecurity IDS
        """
        return message.strip()
    
    def _get_alert_color(self, alert_level: str) -> str:
        """Get color for alert level"""
        colors = {
            'normal': 'good',
            'low': 'warning',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'danger'
        }
        return colors.get(alert_level, 'warning')
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"ALERT_{timestamp}_{len(self.alert_history)}"
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        current_time = datetime.now()
        
        # Count alerts by level
        alert_counts = defaultdict(int)
        recent_alerts = []
        
        for alert in self.alert_history:
            alert_counts[alert['alert_level']] += 1
            
            # Recent alerts (last 24 hours)
            if (current_time - alert['timestamp']) <= timedelta(hours=24):
                recent_alerts.append(alert)
        
        # Calculate rates
        total_alerts = len(self.alert_history)
        recent_count = len(recent_alerts)
        
        # Alert processing rate
        processed_alerts = sum(1 for alert in self.alert_history if alert['processed'])
        processing_rate = processed_alerts / total_alerts if total_alerts > 0 else 0
        
        # Notification success rate
        total_notifications = sum(len(alert['notifications_sent']) for alert in self.alert_history)
        successful_notifications = sum(
            len(alert['notifications_sent']) for alert in self.alert_history if alert['processed']
        )
        notification_success_rate = successful_notifications / total_notifications if total_notifications > 0 else 0
        
        return {
            'total_alerts': total_alerts,
            'recent_alerts_24h': recent_count,
            'alert_counts_by_level': dict(alert_counts),
            'processing_rate': processing_rate,
            'notification_success_rate': notification_success_rate,
            'active_rules': len([rule for rule in self.alert_rules.values() if rule['enabled']]),
            'notification_channels': len([ch for ch in self.notification_channels.values() if ch.get('enabled', False)])
        }
    
    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return list(self.alert_history)[-limit:]
    
    def export_alert_log(self, filepath: str, limit: int = None):
        """Export alert log to file"""
        alerts = list(self.alert_history)
        if limit:
            alerts = alerts[-limit:]
        
        # Convert to JSON-serializable format
        export_data = []
        for alert in alerts:
            export_alert = alert.copy()
            export_alert['timestamp'] = alert['timestamp'].isoformat()
            export_data.append(export_alert)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} alert records to {filepath}")
    
    def clear_alert_history(self):
        """Clear alert history"""
        self.alert_history.clear()
        logger.info("Alert history cleared")
