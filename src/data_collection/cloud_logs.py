"""
Cloud log collection for AWS and Azure environments
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# Optional imports for cloud integration
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.monitor.query import LogsQueryClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import asyncio
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

from ..config.config import CLOUD_CONFIG

logger = logging.getLogger(__name__)


class CloudLogCollector:
    """Collect logs from cloud environments (AWS, Azure)"""
    
    def __init__(self, cloud_provider: str = "aws"):
        self.cloud_provider = cloud_provider.lower()
        self.config = CLOUD_CONFIG.get(self.cloud_provider, {})
        
        if self.cloud_provider == "aws":
            self._init_aws()
        elif self.cloud_provider == "azure":
            self._init_azure()
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")
    
    def _init_aws(self):
        """Initialize AWS clients"""
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not available. AWS integration disabled.")
            self.cloudwatch = None
            self.logs = None
            self.ec2 = None
            return
            
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name=self.config.get('region', 'us-east-1'))
            self.logs = boto3.client('logs', region_name=self.config.get('region', 'us-east-1'))
            self.ec2 = boto3.client('ec2', region_name=self.config.get('region', 'us-east-1'))
            logger.info("AWS clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            self.cloudwatch = None
            self.logs = None
            self.ec2 = None
    
    def _init_azure(self):
        """Initialize Azure clients"""
        if not AZURE_AVAILABLE:
            logger.warning("Azure SDK not available. Azure integration disabled.")
            self.credential = None
            self.logs_client = None
            self.workspace_id = None
            return
            
        try:
            self.credential = DefaultAzureCredential()
            self.logs_client = LogsQueryClient(self.credential)
            self.workspace_id = self.config.get('workspace_id')
            logger.info("Azure clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {str(e)}")
            self.credential = None
            self.logs_client = None
            self.workspace_id = None
    
    def collect_aws_cloudtrail_logs(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Collect AWS CloudTrail logs"""
        if not self.logs:
            logger.warning("AWS logs client not initialized. Returning sample data.")
            return self._generate_sample_cloudtrail_logs(1000)
        
        try:
            # Query CloudTrail logs
            log_group = self.config.get('log_group', 'CloudTrail')
            query = """
            fields @timestamp, eventName, sourceIPAddress, userIdentity.type, 
                   userIdentity.principalId, errorCode, errorMessage
            | filter eventName != "AssumeRole"
            | sort @timestamp desc
            | limit 10000
            """
            
            response = self.logs.start_query(
                logGroupName=log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query
            )
            
            query_id = response['queryId']
            
            # Wait for query to complete
            while True:
                result = self.logs.get_query_results(queryId=query_id)
                if result['status'] == 'Complete':
                    break
                elif result['status'] == 'Failed':
                    raise Exception("Query failed")
                time.sleep(1)
            
            # Process results
            logs_data = []
            for result in result['results']:
                log_entry = {}
                for field in result:
                    log_entry[field['field']] = field['value']
                logs_data.append(log_entry)
            
            return pd.DataFrame(logs_data)
            
        except Exception as e:
            logger.error(f"Error collecting AWS CloudTrail logs: {str(e)}")
            return self._generate_sample_cloudtrail_logs(1000)
    
    def collect_aws_vpc_flow_logs(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Collect AWS VPC Flow logs"""
        if not self.logs:
            logger.warning("AWS logs client not initialized. Returning sample data.")
            return self._generate_sample_vpc_flow_logs(1000)
        
        try:
            # Query VPC Flow logs
            log_group = "VPCFlowLogs"
            query = """
            fields @timestamp, srcaddr, dstaddr, srcport, dstport, protocol, 
                   packets, bytes, action, log-status
            | filter action = "ACCEPT" or action = "REJECT"
            | sort @timestamp desc
            | limit 10000
            """
            
            response = self.logs.start_query(
                logGroupName=log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query
            )
            
            query_id = response['queryId']
            
            # Wait for query to complete
            while True:
                result = self.logs.get_query_results(queryId=query_id)
                if result['status'] == 'Complete':
                    break
                elif result['status'] == 'Failed':
                    raise Exception("Query failed")
                time.sleep(1)
            
            # Process results
            logs_data = []
            for result in result['results']:
                log_entry = {}
                for field in result:
                    log_entry[field['field']] = field['value']
                logs_data.append(log_entry)
            
            return pd.DataFrame(logs_data)
            
        except Exception as e:
            logger.error(f"Error collecting AWS VPC Flow logs: {str(e)}")
            return self._generate_sample_vpc_flow_logs(1000)
    
    def collect_azure_activity_logs(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Collect Azure Activity logs"""
        if not self.logs_client or not self.workspace_id:
            logger.warning("Azure logs client not initialized. Returning sample data.")
            return self._generate_sample_azure_logs(1000)
        
        try:
            # KQL query for Azure Activity logs
            query = """
            AzureActivity
            | where TimeGenerated >= datetime({start_time}) and TimeGenerated <= datetime({end_time})
            | project TimeGenerated, ActivityStatus, Caller, OperationName, ResourceGroup, Resource
            | order by TimeGenerated desc
            | take 10000
            """.format(
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
            
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timedelta(hours=24)
            )
            
            # Convert to DataFrame
            logs_data = []
            for table in response.tables:
                for row in table.rows:
                    log_entry = {}
                    for i, column in enumerate(table.columns):
                        log_entry[column.name] = row[i]
                    logs_data.append(log_entry)
            
            return pd.DataFrame(logs_data)
            
        except Exception as e:
            logger.error(f"Error collecting Azure Activity logs: {str(e)}")
            return self._generate_sample_azure_logs(1000)
    
    def collect_azure_network_logs(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Collect Azure Network Security Group logs"""
        if not self.logs_client or not self.workspace_id:
            logger.warning("Azure logs client not initialized. Returning sample data.")
            return self._generate_sample_azure_network_logs(1000)
        
        try:
            # KQL query for NSG logs
            query = """
            AzureDiagnostics
            | where Category == "NetworkSecurityGroupRuleCounter"
            | where TimeGenerated >= datetime({start_time}) and TimeGenerated <= datetime({end_time})
            | project TimeGenerated, SourceIP, DestinationIP, SourcePort, DestinationPort, 
                     Protocol, Action, Rule
            | order by TimeGenerated desc
            | take 10000
            """.format(
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
            
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timedelta(hours=24)
            )
            
            # Convert to DataFrame
            logs_data = []
            for table in response.tables:
                for row in table.rows:
                    log_entry = {}
                    for i, column in enumerate(table.columns):
                        log_entry[column.name] = row[i]
                    logs_data.append(log_entry)
            
            return pd.DataFrame(logs_data)
            
        except Exception as e:
            logger.error(f"Error collecting Azure Network logs: {str(e)}")
            return self._generate_sample_azure_network_logs(1000)
    
    def _generate_sample_cloudtrail_logs(self, n_samples: int) -> pd.DataFrame:
        """Generate sample CloudTrail logs for testing"""
        np.random.seed(42)
        
        events = [
            'DescribeInstances', 'RunInstances', 'TerminateInstances', 'CreateSecurityGroup',
            'AuthorizeSecurityGroupIngress', 'RevokeSecurityGroupIngress', 'CreateBucket',
            'PutObject', 'GetObject', 'DeleteObject', 'AssumeRole', 'GetCallerIdentity'
        ]
        
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='1min'),
            'eventName': np.random.choice(events, n_samples),
            'sourceIPAddress': [f"203.0.113.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'userIdentity.type': np.random.choice(['IAMUser', 'AssumedRole', 'Root'], n_samples),
            'userIdentity.principalId': [f"user_{np.random.randint(1000,9999)}" for _ in range(n_samples)],
            'errorCode': np.random.choice(['', 'AccessDenied', 'UnauthorizedOperation'], n_samples, p=[0.8, 0.15, 0.05]),
            'errorMessage': np.random.choice(['', 'User is not authorized', 'Access denied'], n_samples, p=[0.8, 0.1, 0.1])
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_vpc_flow_logs(self, n_samples: int) -> pd.DataFrame:
        """Generate sample VPC Flow logs for testing"""
        np.random.seed(42)
        
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='1min'),
            'srcaddr': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'dstaddr': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'srcport': np.random.randint(1, 65535, n_samples),
            'dstport': np.random.randint(1, 65535, n_samples),
            'protocol': np.random.choice([6, 17, 1], n_samples),  # TCP, UDP, ICMP
            'packets': np.random.poisson(10, n_samples),
            'bytes': np.random.poisson(1000, n_samples),
            'action': np.random.choice(['ACCEPT', 'REJECT'], n_samples, p=[0.9, 0.1]),
            'log-status': np.random.choice(['OK', 'NODATA', 'SKIPDATA'], n_samples, p=[0.95, 0.03, 0.02])
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_azure_logs(self, n_samples: int) -> pd.DataFrame:
        """Generate sample Azure Activity logs for testing"""
        np.random.seed(42)
        
        operations = [
            'Microsoft.Compute/virtualMachines/read',
            'Microsoft.Compute/virtualMachines/write',
            'Microsoft.Network/networkSecurityGroups/read',
            'Microsoft.Storage/storageAccounts/read',
            'Microsoft.Storage/storageAccounts/write'
        ]
        
        data = {
            'TimeGenerated': pd.date_range(start='2023-01-01', periods=n_samples, freq='1min'),
            'ActivityStatus': np.random.choice(['Success', 'Failed'], n_samples, p=[0.9, 0.1]),
            'Caller': [f"user_{np.random.randint(1000,9999)}@company.com" for _ in range(n_samples)],
            'OperationName': np.random.choice(operations, n_samples),
            'ResourceGroup': [f"rg-{np.random.randint(1,100)}" for _ in range(n_samples)],
            'Resource': [f"resource-{np.random.randint(1,1000)}" for _ in range(n_samples)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_azure_network_logs(self, n_samples: int) -> pd.DataFrame:
        """Generate sample Azure Network logs for testing"""
        np.random.seed(42)
        
        data = {
            'TimeGenerated': pd.date_range(start='2023-01-01', periods=n_samples, freq='1min'),
            'SourceIP': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'DestinationIP': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'SourcePort': np.random.randint(1, 65535, n_samples),
            'DestinationPort': np.random.randint(1, 65535, n_samples),
            'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
            'Action': np.random.choice(['Allow', 'Deny'], n_samples, p=[0.8, 0.2]),
            'Rule': [f"NSG-{np.random.randint(1,100)}" for _ in range(n_samples)]
        }
        
        return pd.DataFrame(data)
    
    def collect_all_logs(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Collect all available logs from the cloud provider"""
        logs = {}
        
        if self.cloud_provider == "aws":
            logs['cloudtrail'] = self.collect_aws_cloudtrail_logs(start_time, end_time)
            logs['vpc_flow'] = self.collect_aws_vpc_flow_logs(start_time, end_time)
        elif self.cloud_provider == "azure":
            logs['activity'] = self.collect_azure_activity_logs(start_time, end_time)
            logs['network'] = self.collect_azure_network_logs(start_time, end_time)
        
        return logs
