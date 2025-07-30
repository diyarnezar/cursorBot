#!/usr/bin/env python3
"""
🌐 Network Diagnostic Tool for Project Hyperion
==============================================

Comprehensive network connectivity testing for Binance API and related services.
Helps diagnose and troubleshoot network connectivity issues.

Author: Project Hyperion Team
Date: 2025
"""

import sys
import time
import socket
import requests
import dns.resolver
import subprocess
import platform
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure basic logging without conflicts
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkDiagnostic:
    """Comprehensive network diagnostic tool"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def run_comprehensive_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive network diagnostic"""
        logger.info("🌐 Starting comprehensive network diagnostic...")
        
        # System information
        self.results['system_info'] = self.get_system_info()
        
        # Basic connectivity tests
        self.results['basic_connectivity'] = self.test_basic_connectivity()
        
        # DNS resolution tests
        self.results['dns_tests'] = self.test_dns_resolution()
        
        # Network route tests
        self.results['route_tests'] = self.test_network_routes()
        
        # API connectivity tests
        self.results['api_tests'] = self.test_api_connectivity()
        
        # Firewall and proxy tests
        self.results['firewall_tests'] = self.test_firewall_proxy()
        
        # Performance tests
        self.results['performance_tests'] = self.test_network_performance()
        
        # Summary and recommendations
        self.results['summary'] = self.generate_summary()
        
        return self.results
    
    def get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        logger.info("📋 Collecting system information...")
        
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ System info collected: {info['platform']}")
        return info
    
    def test_basic_connectivity(self) -> Dict[str, bool]:
        """Test basic internet connectivity"""
        logger.info("🌍 Testing basic internet connectivity...")
        
        results = {}
        
        # Test common DNS servers
        dns_servers = ['8.8.8.8', '1.1.1.1', '208.67.222.222']
        for dns in dns_servers:
            try:
                socket.create_connection((dns, 53), timeout=5)
                results[f'dns_{dns}'] = True
                logger.info(f"✅ DNS server {dns} reachable")
            except Exception as e:
                results[f'dns_{dns}'] = False
                logger.error(f"❌ DNS server {dns} not reachable: {e}")
        
        # Test common websites
        websites = ['google.com', 'cloudflare.com', 'amazon.com']
        for site in websites:
            try:
                socket.gethostbyname(site)
                results[f'website_{site}'] = True
                logger.info(f"✅ Website {site} resolvable")
            except Exception as e:
                results[f'website_{site}'] = False
                logger.error(f"❌ Website {site} not resolvable: {e}")
        
        return results
    
    def test_dns_resolution(self) -> Dict[str, Any]:
        """Test DNS resolution for Binance and related services"""
        logger.info("🔍 Testing DNS resolution...")
        
        results = {}
        
        # Test Binance domains
        binance_domains = [
            'api.binance.com',
            'stream.binance.com',
            'fapi.binance.com',
            'dapi.binance.com'
        ]
        
        for domain in binance_domains:
            try:
                logger.info(f"🔍 Resolving {domain}...")
                resolver = dns.resolver.Resolver()
                resolver.timeout = 10
                resolver.lifetime = 10
                
                answers = resolver.resolve(domain, 'A')
                ip_addresses = [str(answer) for answer in answers]
                
                results[domain] = {
                    'resolved': True,
                    'ip_addresses': ip_addresses,
                    'ttl': answers.rrset.ttl if hasattr(answers, 'rrset') else None
                }
                logger.info(f"✅ {domain} -> {ip_addresses}")
                
            except dns.resolver.NXDOMAIN:
                results[domain] = {'resolved': False, 'error': 'NXDOMAIN'}
                logger.error(f"❌ {domain}: Domain does not exist")
            except dns.resolver.Timeout:
                results[domain] = {'resolved': False, 'error': 'Timeout'}
                logger.error(f"❌ {domain}: DNS resolution timeout")
            except Exception as e:
                results[domain] = {'resolved': False, 'error': str(e)}
                logger.error(f"❌ {domain}: {e}")
        
        return results
    
    def test_network_routes(self) -> Dict[str, Any]:
        """Test network routes to Binance servers"""
        logger.info("🛣️ Testing network routes...")
        
        results = {}
        
        # Test traceroute to Binance API
        try:
            if platform.system() == "Windows":
                cmd = ['tracert', 'api.binance.com']
            else:
                cmd = ['traceroute', 'api.binance.com']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                results['traceroute'] = {
                    'success': True,
                    'output': result.stdout,
                    'error': None
                }
                logger.info("✅ Traceroute to api.binance.com successful")
            else:
                results['traceroute'] = {
                    'success': False,
                    'output': result.stdout,
                    'error': result.stderr
                }
                logger.warning(f"⚠️ Traceroute failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            results['traceroute'] = {
                'success': False,
                'output': None,
                'error': 'Timeout'
            }
            logger.error("❌ Traceroute timeout")
        except Exception as e:
            results['traceroute'] = {
                'success': False,
                'output': None,
                'error': str(e)
            }
            logger.error(f"❌ Traceroute error: {e}")
        
        return results
    
    def test_api_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to Binance API endpoints"""
        logger.info("🔌 Testing API connectivity...")
        
        results = {}
        
        # Test different Binance API endpoints
        endpoints = [
            'https://api.binance.com/api/v3/ping',
            'https://api.binance.com/api/v3/time',
            'https://api.binance.com/api/v3/exchangeInfo'
        ]
        
        for endpoint in endpoints:
            try:
                logger.info(f"🔌 Testing {endpoint}...")
                response = requests.get(endpoint, timeout=10)
                
                results[endpoint] = {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'success': response.status_code == 200,
                    'content_length': len(response.content)
                }
                
                if response.status_code == 200:
                    logger.info(f"✅ {endpoint}: {response.status_code} ({response.elapsed.total_seconds():.2f}s)")
                else:
                    logger.warning(f"⚠️ {endpoint}: {response.status_code}")
                    
            except requests.exceptions.ConnectionError as e:
                results[endpoint] = {
                    'status_code': None,
                    'response_time': None,
                    'success': False,
                    'error': f'ConnectionError: {e}'
                }
                logger.error(f"❌ {endpoint}: Connection error - {e}")
            except requests.exceptions.Timeout as e:
                results[endpoint] = {
                    'status_code': None,
                    'response_time': None,
                    'success': False,
                    'error': f'Timeout: {e}'
                }
                logger.error(f"❌ {endpoint}: Timeout - {e}")
            except Exception as e:
                results[endpoint] = {
                    'status_code': None,
                    'response_time': None,
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"❌ {endpoint}: {e}")
        
        return results
    
    def test_firewall_proxy(self) -> Dict[str, Any]:
        """Test for firewall and proxy issues"""
        logger.info("🛡️ Testing firewall and proxy...")
        
        results = {}
        
        # Check for proxy environment variables
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        for var in proxy_vars:
            value = os.environ.get(var)
            results[f'proxy_{var}'] = value if value else None
        
        # Test direct connection vs proxy
        try:
            # Test without proxy
            session = requests.Session()
            session.proxies.clear()
            
            response = session.get('https://api.binance.com/api/v3/ping', timeout=10)
            results['direct_connection'] = response.status_code == 200
            
        except Exception as e:
            results['direct_connection'] = False
            results['direct_connection_error'] = str(e)
        
        return results
    
    def test_network_performance(self) -> Dict[str, Any]:
        """Test network performance metrics"""
        logger.info("⚡ Testing network performance...")
        
        results = {}
        
        # Test latency to Binance API
        latencies = []
        for i in range(5):
            try:
                start_time = time.time()
                response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
                end_time = time.time()
                
                if response.status_code == 200:
                    latency = (end_time - start_time) * 1000  # Convert to milliseconds
                    latencies.append(latency)
                    logger.info(f"📊 Ping {i+1}: {latency:.2f}ms")
                else:
                    logger.warning(f"⚠️ Ping {i+1}: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Ping {i+1}: {e}")
        
        if latencies:
            results['latency'] = {
                'min': min(latencies),
                'max': max(latencies),
                'avg': sum(latencies) / len(latencies),
                'count': len(latencies)
            }
            logger.info(f"📊 Latency stats: min={min(latencies):.2f}ms, max={max(latencies):.2f}ms, avg={sum(latencies)/len(latencies):.2f}ms")
        else:
            results['latency'] = None
            logger.error("❌ No successful latency measurements")
        
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate diagnostic summary and recommendations"""
        logger.info("📋 Generating diagnostic summary...")
        
        summary = {
            'overall_status': 'unknown',
            'issues_found': [],
            'recommendations': [],
            'diagnostic_duration': (datetime.now() - self.start_time).total_seconds()
        }
        
        # Check DNS resolution
        dns_tests = self.results.get('dns_tests', {})
        dns_failures = [domain for domain, result in dns_tests.items() 
                       if not result.get('resolved', False)]
        
        if dns_failures:
            summary['issues_found'].append(f"DNS resolution failed for: {', '.join(dns_failures)}")
            summary['recommendations'].append("Check DNS settings and try using different DNS servers (8.8.8.8, 1.1.1.1)")
        
        # Check API connectivity
        api_tests = self.results.get('api_tests', {})
        api_failures = [endpoint for endpoint, result in api_tests.items() 
                       if not result.get('success', False)]
        
        if api_failures:
            summary['issues_found'].append(f"API connectivity failed for: {', '.join(api_failures)}")
            summary['recommendations'].append("Check firewall settings and ensure outbound HTTPS traffic is allowed")
        
        # Check basic connectivity
        basic_tests = self.results.get('basic_connectivity', {})
        basic_failures = [test for test, result in basic_tests.items() if not result]
        
        if basic_failures:
            summary['issues_found'].append(f"Basic connectivity failed for: {', '.join(basic_failures)}")
            summary['recommendations'].append("Check internet connection and network configuration")
        
        # Determine overall status
        if not summary['issues_found']:
            summary['overall_status'] = 'healthy'
            summary['recommendations'].append("Network connectivity appears healthy")
        elif len(summary['issues_found']) <= 2:
            summary['overall_status'] = 'degraded'
        else:
            summary['overall_status'] = 'unhealthy'
        
        return summary
    
    def print_report(self):
        """Print formatted diagnostic report"""
        print("\n" + "="*80)
        print("🌐 NETWORK DIAGNOSTIC REPORT")
        print("="*80)
        print(f"📅 Timestamp: {self.results['system_info']['timestamp']}")
        print(f"💻 System: {self.results['system_info']['platform']}")
        print(f"🐍 Python: {self.results['system_info']['python_version'].split()[0]}")
        
        # Summary
        summary = self.results['summary']
        print(f"\n📊 OVERALL STATUS: {summary['overall_status'].upper()}")
        print(f"⏱️ Diagnostic Duration: {summary['diagnostic_duration']:.2f} seconds")
        
        if summary['issues_found']:
            print(f"\n❌ ISSUES FOUND:")
            for issue in summary['issues_found']:
                print(f"   • {issue}")
        
        if summary['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        # Detailed results
        print(f"\n🔍 DETAILED RESULTS:")
        
        # DNS Results
        print(f"\n🔍 DNS Resolution:")
        for domain, result in self.results['dns_tests'].items():
            status = "✅" if result.get('resolved') else "❌"
            print(f"   {status} {domain}")
            if result.get('resolved'):
                print(f"      IPs: {', '.join(result['ip_addresses'])}")
            else:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        # API Results
        print(f"\n🔌 API Connectivity:")
        for endpoint, result in self.results['api_tests'].items():
            status = "✅" if result.get('success') else "❌"
            print(f"   {status} {endpoint}")
            if result.get('success'):
                print(f"      Status: {result['status_code']}, Time: {result['response_time']:.2f}s")
            else:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        # Performance Results
        if self.results['performance_tests'].get('latency'):
            latency = self.results['performance_tests']['latency']
            print(f"\n⚡ Network Performance:")
            print(f"   📊 Latency: {latency['avg']:.2f}ms avg ({latency['min']:.2f}ms min, {latency['max']:.2f}ms max)")
        
        print("\n" + "="*80)


def main():
    """Main diagnostic function"""
    print("🌐 Project Hyperion - Network Diagnostic Tool")
    print("="*50)
    
    try:
        # Run diagnostic
        diagnostic = NetworkDiagnostic()
        results = diagnostic.run_comprehensive_diagnostic()
        
        # Print report
        diagnostic.print_report()
        
        # Save results to file
        import json
        with open('network_diagnostic_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: network_diagnostic_results.json")
        
        # Return appropriate exit code
        summary = results['summary']
        if summary['overall_status'] == 'healthy':
            print("\n✅ Network connectivity appears healthy!")
            return 0
        elif summary['overall_status'] == 'degraded':
            print("\n⚠️ Network connectivity is degraded. Check recommendations above.")
            return 1
        else:
            print("\n❌ Network connectivity issues detected. Check recommendations above.")
            return 2
            
    except KeyboardInterrupt:
        print("\n\n🛑 Diagnostic interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Diagnostic error: {e}")
        logger.error(f"Diagnostic error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 