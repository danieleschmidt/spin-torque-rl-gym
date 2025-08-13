#!/usr/bin/env python3
"""
Automated metrics collection script for Spin-Torque RL-Gym.

This script collects metrics from various sources and updates the project metrics.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class MetricsCollector:
    """Collects and aggregates project metrics from various sources."""

    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize the metrics collector.
        
        Args:
            config_path: Path to the project metrics configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo = self.config["project"]["repository"]

    def _load_config(self) -> Dict[str, Any]:
        """Load the metrics configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Metrics config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return json.load(f)

    def _save_config(self) -> None:
        """Save the updated metrics configuration."""
        self.config["project"]["last_updated"] = datetime.now(timezone.utc).isoformat()

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def _github_api_request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make a request to the GitHub API.
        
        Args:
            endpoint: API endpoint (e.g., 'repos/owner/repo')
            
        Returns:
            JSON response or None if request failed
        """
        if not self.github_token:
            print("Warning: GITHUB_TOKEN not set, skipping GitHub metrics")
            return None

        url = f"https://api.github.com/{endpoint}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"GitHub API request failed: {e}")
            return None

    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        print("Collecting GitHub metrics...")

        metrics = {}

        # Repository basic info
        repo_data = self._github_api_request(f"repos/{self.repo}")
        if repo_data:
            metrics["github_stars"] = repo_data.get("stargazers_count", 0)
            metrics["github_forks"] = repo_data.get("forks_count", 0)
            metrics["github_watchers"] = repo_data.get("watchers_count", 0)
            metrics["github_open_issues"] = repo_data.get("open_issues_count", 0)
            metrics["github_size"] = repo_data.get("size", 0)  # KB

        # Contributors
        contributors_data = self._github_api_request(f"repos/{self.repo}/contributors")
        if contributors_data:
            metrics["contributors"] = len(contributors_data)

        # Recent commits (last week)
        commits_data = self._github_api_request(
            f"repos/{self.repo}/commits?since={(datetime.now() - timedelta(days=7)).isoformat()}"
        )
        if commits_data:
            metrics["commits_last_week"] = len(commits_data)

        # Pull requests
        prs_data = self._github_api_request(f"repos/{self.repo}/pulls?state=all")
        if prs_data:
            open_prs = [pr for pr in prs_data if pr["state"] == "open"]
            closed_prs = [pr for pr in prs_data if pr["state"] == "closed"]
            metrics["open_pull_requests"] = len(open_prs)
            metrics["closed_pull_requests"] = len(closed_prs)

        # Issues
        issues_data = self._github_api_request(f"repos/{self.repo}/issues?state=all")
        if issues_data:
            # Filter out pull requests (GitHub API includes PRs as issues)
            actual_issues = [issue for issue in issues_data if not issue.get("pull_request")]
            open_issues = [issue for issue in actual_issues if issue["state"] == "open"]
            closed_issues = [issue for issue in actual_issues if issue["state"] == "closed"]
            metrics["open_issues"] = len(open_issues)
            metrics["closed_issues"] = len(closed_issues)

        # Releases
        releases_data = self._github_api_request(f"repos/{self.repo}/releases")
        if releases_data:
            metrics["total_releases"] = len(releases_data)
            if releases_data:
                latest_release = releases_data[0]
                metrics["latest_release"] = {
                    "tag": latest_release.get("tag_name"),
                    "published_at": latest_release.get("published_at"),
                    "downloads": sum(
                        asset.get("download_count", 0)
                        for asset in latest_release.get("assets", [])
                    )
                }

        # Languages
        languages_data = self._github_api_request(f"repos/{self.repo}/languages")
        if languages_data:
            total_bytes = sum(languages_data.values())
            metrics["languages"] = {
                lang: round((bytes_count / total_bytes) * 100, 2)
                for lang, bytes_count in languages_data.items()
            }

        return metrics

    def collect_ci_metrics(self) -> Dict[str, Any]:
        """Collect CI/CD metrics from GitHub Actions."""
        print("Collecting CI/CD metrics...")

        metrics = {}

        # Workflow runs
        runs_data = self._github_api_request(f"repos/{self.repo}/actions/runs?per_page=50")
        if runs_data and "workflow_runs" in runs_data:
            runs = runs_data["workflow_runs"]

            successful_runs = [run for run in runs if run["conclusion"] == "success"]
            failed_runs = [run for run in runs if run["conclusion"] == "failure"]

            metrics["workflow_success_rate"] = (
                len(successful_runs) / len(runs) * 100 if runs else 0
            )
            metrics["total_workflow_runs"] = len(runs)
            metrics["successful_runs"] = len(successful_runs)
            metrics["failed_runs"] = len(failed_runs)

            # Average build time (for completed runs)
            completed_runs = [
                run for run in runs
                if run["conclusion"] in ["success", "failure"] and run["updated_at"] and run["created_at"]
            ]

            if completed_runs:
                build_times = []
                for run in completed_runs[:10]:  # Last 10 runs
                    try:
                        created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                        completed = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                        build_time = (completed - created).total_seconds()
                        build_times.append(build_time)
                    except (ValueError, TypeError):
                        continue

                if build_times:
                    metrics["average_build_time"] = sum(build_times) / len(build_times)
                    metrics["max_build_time"] = max(build_times)
                    metrics["min_build_time"] = min(build_times)

        return metrics

    def collect_package_metrics(self) -> Dict[str, Any]:
        """Collect package-related metrics."""
        print("Collecting package metrics...")

        metrics = {}

        # PyPI downloads (if package is published)
        package_name = self.config["project"]["name"].lower().replace(" ", "-")
        try:
            response = requests.get(
                f"https://pypistats.org/api/packages/{package_name}/recent",
                timeout=10
            )
            if response.status_code == 200:
                pypi_data = response.json()
                metrics["pypi_downloads_last_day"] = pypi_data.get("data", {}).get("last_day", 0)
                metrics["pypi_downloads_last_week"] = pypi_data.get("data", {}).get("last_week", 0)
                metrics["pypi_downloads_last_month"] = pypi_data.get("data", {}).get("last_month", 0)
        except requests.RequestException:
            print("Warning: Could not fetch PyPI download stats")

        # Docker Hub metrics (if applicable)
        # This would require Docker Hub API integration

        return metrics

    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics from static analysis."""
        print("Collecting code quality metrics...")

        metrics = {}

        # This would typically integrate with tools like:
        # - SonarQube
        # - CodeClimate
        # - Codecov
        # - etc.

        # For now, we'll collect basic file statistics
        try:
            repo_root = Path(__file__).parent.parent
            python_files = list(repo_root.rglob("*.py"))

            metrics["total_python_files"] = len(python_files)

            total_lines = 0
            total_blank_lines = 0
            total_comment_lines = 0

            for file_path in python_files:
                if "/.git/" in str(file_path) or "__pycache__" in str(file_path):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        total_lines += len(lines)

                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                total_blank_lines += 1
                            elif stripped.startswith("#"):
                                total_comment_lines += 1
                except (UnicodeDecodeError, IOError):
                    continue

            metrics["total_lines_of_code"] = total_lines
            metrics["blank_lines"] = total_blank_lines
            metrics["comment_lines"] = total_comment_lines
            metrics["code_lines"] = total_lines - total_blank_lines - total_comment_lines

            if total_lines > 0:
                metrics["comment_ratio"] = round((total_comment_lines / total_lines) * 100, 2)

        except Exception as e:
            print(f"Warning: Could not collect code metrics: {e}")

        return metrics

    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        print("Collecting security metrics...")

        metrics = {}

        # This would typically integrate with:
        # - GitHub Security Advisories
        # - Snyk
        # - OWASP Dependency Check
        # - Bandit results
        # etc.

        # Check for security-related files
        repo_root = Path(__file__).parent.parent
        security_files = [
            "SECURITY.md",
            ".github/dependabot.yml",
            ".github/workflows/security-scan.yml",
            "bandit.yml",
            ".pre-commit-config.yaml"
        ]

        metrics["security_files_present"] = sum(
            1 for file in security_files if (repo_root / file).exists()
        )
        metrics["total_security_files"] = len(security_files)
        metrics["security_files_coverage"] = round(
            (metrics["security_files_present"] / metrics["total_security_files"]) * 100, 2
        )

        return metrics

    def update_metrics(self, collected_metrics: Dict[str, Any]) -> None:
        """Update the metrics configuration with collected data."""
        print("Updating metrics configuration...")

        timestamp = datetime.now(timezone.utc).isoformat()

        # Map collected metrics to configuration structure
        metric_mappings = {
            # GitHub metrics
            "github_stars": "business.adoption.github_stars",
            "github_forks": "business.adoption.github_forks",
            "contributors": "project_health.collaboration.contributors",
            "commits_last_week": "project_health.activity.commits_per_week",

            # CI/CD metrics
            "workflow_success_rate": "operational.deployment.deployment_success_rate",
            "average_build_time": "development.performance.build_time",

            # Package metrics
            "pypi_downloads_last_month": "business.adoption.downloads",

            # Code quality metrics
            "total_lines_of_code": "development.code_quality.lines_of_code",
            "comment_ratio": "development.code_quality.documentation_coverage",

            # Security metrics
            "security_files_coverage": "development.security.configuration_coverage"
        }

        for metric_key, config_path in metric_mappings.items():
            if metric_key in collected_metrics:
                value = collected_metrics[metric_key]

                # Navigate to the nested configuration
                path_parts = config_path.split(".")
                config_section = self.config["metrics"]

                for part in path_parts[:-1]:
                    if part not in config_section:
                        config_section[part] = {}
                    config_section = config_section[part]

                final_key = path_parts[-1]
                if final_key in config_section:
                    config_section[final_key]["current"] = value
                    config_section[final_key]["last_measured"] = timestamp

                    # Simple trend analysis
                    if "previous" in config_section[final_key]:
                        previous = config_section[final_key]["previous"]
                        if value > previous:
                            config_section[final_key]["trend"] = "growing"
                        elif value < previous:
                            config_section[final_key]["trend"] = "declining"
                        else:
                            config_section[final_key]["trend"] = "stable"

                    config_section[final_key]["previous"] = value

    def generate_summary_report(self, collected_metrics: Dict[str, Any]) -> str:
        """Generate a summary report of collected metrics."""
        report_lines = [
            "# Metrics Collection Summary",
            f"Generated at: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Key Metrics",
            ""
        ]

        # Key metrics to highlight
        key_metrics = [
            ("GitHub Stars", collected_metrics.get("github_stars", 0)),
            ("GitHub Forks", collected_metrics.get("github_forks", 0)),
            ("Contributors", collected_metrics.get("contributors", 0)),
            ("Open Issues", collected_metrics.get("open_issues", 0)),
            ("Workflow Success Rate", f"{collected_metrics.get('workflow_success_rate', 0):.1f}%"),
            ("Average Build Time", f"{collected_metrics.get('average_build_time', 0):.1f}s"),
            ("Lines of Code", collected_metrics.get("total_lines_of_code", 0)),
            ("Comment Ratio", f"{collected_metrics.get('comment_ratio', 0):.1f}%"),
        ]

        for metric_name, value in key_metrics:
            report_lines.append(f"- **{metric_name}**: {value}")

        report_lines.extend([
            "",
            "## Collection Status",
            f"- Total metrics collected: {len(collected_metrics)}",
            f"- GitHub API status: {'✅ Connected' if self.github_token else '❌ No token'}",
            "",
            "## Trends",
            ""
        ])

        # Add trend information
        trends = self._analyze_trends(collected_metrics)
        for trend in trends:
            report_lines.append(f"- {trend}")

        return "\n".join(report_lines)

    def _analyze_trends(self, collected_metrics: Dict[str, Any]) -> list:
        """Analyze trends in collected metrics."""
        trends = []

        # Simple trend analysis based on current vs target
        github_stars = collected_metrics.get("github_stars", 0)
        target_stars = self.config["metrics"]["business"]["adoption"]["github_stars"]["target"]

        if github_stars >= target_stars:
            trends.append("🎯 GitHub stars target achieved!")
        elif github_stars > 0:
            progress = (github_stars / target_stars) * 100
            trends.append(f"📈 GitHub stars at {progress:.1f}% of target")

        # Add more trend analysis as needed

        return trends

    def run(self) -> None:
        """Run the complete metrics collection process."""
        print("Starting metrics collection...")

        all_metrics = {}

        # Collect from different sources
        try:
            github_metrics = self.collect_github_metrics()
            all_metrics.update(github_metrics)
        except Exception as e:
            print(f"Error collecting GitHub metrics: {e}")

        try:
            ci_metrics = self.collect_ci_metrics()
            all_metrics.update(ci_metrics)
        except Exception as e:
            print(f"Error collecting CI metrics: {e}")

        try:
            package_metrics = self.collect_package_metrics()
            all_metrics.update(package_metrics)
        except Exception as e:
            print(f"Error collecting package metrics: {e}")

        try:
            quality_metrics = self.collect_code_quality_metrics()
            all_metrics.update(quality_metrics)
        except Exception as e:
            print(f"Error collecting code quality metrics: {e}")

        try:
            security_metrics = self.collect_security_metrics()
            all_metrics.update(security_metrics)
        except Exception as e:
            print(f"Error collecting security metrics: {e}")

        # Update configuration
        self.update_metrics(all_metrics)
        self._save_config()

        # Generate and display summary
        summary = self.generate_summary_report(all_metrics)
        print("\n" + "="*60)
        print(summary)
        print("="*60)

        # Save summary to file
        summary_path = Path("metrics-summary.md")
        with open(summary_path, "w") as f:
            f.write(summary)

        print(f"\nMetrics collection completed. Summary saved to {summary_path}")


def main():
    """Main entry point for the metrics collection script."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument(
        "--config",
        default=".github/project-metrics.json",
        help="Path to metrics configuration file"
    )
    parser.add_argument(
        "--output",
        help="Output file for summary report"
    )
    parser.add_argument(
        "--github-only",
        action="store_true",
        help="Collect only GitHub metrics"
    )

    args = parser.parse_args()

    try:
        collector = MetricsCollector(args.config)

        if args.github_only:
            print("Collecting GitHub metrics only...")
            metrics = collector.collect_github_metrics()
            collector.update_metrics(metrics)
            collector._save_config()
        else:
            collector.run()

        if args.output:
            summary = collector.generate_summary_report({})
            with open(args.output, "w") as f:
                f.write(summary)
            print(f"Summary written to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
