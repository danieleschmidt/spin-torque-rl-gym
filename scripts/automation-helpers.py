#!/usr/bin/env python3
"""
Automation helper scripts for Spin-Torque RL-Gym repository maintenance.

This module provides various automation utilities for repository maintenance,
code quality checks, and development workflow optimization.
"""

import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests


class RepositoryMaintainer:
    """Handles automated repository maintenance tasks."""

    def __init__(self, repo_path: str = "."):
        """Initialize the repository maintainer.
        
        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")

    def cleanup_old_branches(self, days_old: int = 30) -> List[str]:
        """Clean up old merged branches.
        
        Args:
            days_old: Remove branches older than this many days
            
        Returns:
            List of deleted branch names
        """
        print(f"Cleaning up branches older than {days_old} days...")

        try:
            # Get all remote branches with their last commit dates
            result = subprocess.run([
                "git", "for-each-ref",
                "--format=%(refname:short) %(committerdate:iso8601)",
                "refs/remotes/origin"
            ], cwd=self.repo_path, capture_output=True, text=True, check=True)

            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_branches = []

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue

                branch_name = parts[0].replace("origin/", "")
                commit_date_str = parts[1]

                # Skip main branches
                if branch_name in ["main", "develop", "master"]:
                    continue

                try:
                    commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
                    if commit_date < cutoff_date:
                        print(f"Deleting old branch: {branch_name}")
                        subprocess.run([
                            "git", "push", "origin", "--delete", branch_name
                        ], cwd=self.repo_path, check=True)
                        deleted_branches.append(branch_name)
                except (ValueError, subprocess.CalledProcessError) as e:
                    print(f"Warning: Could not process branch {branch_name}: {e}")

            return deleted_branches

        except subprocess.CalledProcessError as e:
            print(f"Error cleaning up branches: {e}")
            return []

    def update_dependencies(self, dependency_type: str = "all") -> bool:
        """Update project dependencies.
        
        Args:
            dependency_type: Type of dependencies to update (all, security, minor, patch)
            
        Returns:
            True if updates were successful
        """
        print(f"Updating {dependency_type} dependencies...")

        try:
            # Install pip-upgrader if not available
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pip-upgrader"
            ], check=True, capture_output=True)

            # Update dependencies based on type
            if dependency_type == "security":
                # Use safety to identify security issues
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "safety"
                ], check=True, capture_output=True)

                result = subprocess.run([
                    "safety", "check", "--json"
                ], check=False, capture_output=True, text=True)

                if result.returncode != 0:
                    print("Security vulnerabilities found:")
                    print(result.stdout)
                    return False

            elif dependency_type in ["minor", "patch", "all"]:
                # Use pip-upgrader for safe updates
                upgrade_args = ["pip-upgrade"]

                if dependency_type == "patch":
                    upgrade_args.append("--patch-only")
                elif dependency_type == "minor":
                    upgrade_args.append("--minor-only")

                subprocess.run(upgrade_args, cwd=self.repo_path, check=True)

            print(f"Dependencies updated successfully ({dependency_type})")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error updating dependencies: {e}")
            return False

    def generate_changelog(self, since_tag: Optional[str] = None) -> str:
        """Generate changelog from git commits.
        
        Args:
            since_tag: Generate changelog since this tag (latest if None)
            
        Returns:
            Generated changelog content
        """
        print("Generating changelog...")

        try:
            # Get the range for changelog
            if since_tag:
                commit_range = f"{since_tag}..HEAD"
            else:
                # Get latest tag
                result = subprocess.run([
                    "git", "describe", "--tags", "--abbrev=0"
                ], check=False, cwd=self.repo_path, capture_output=True, text=True)

                if result.returncode == 0:
                    latest_tag = result.stdout.strip()
                    commit_range = f"{latest_tag}..HEAD"
                else:
                    commit_range = "HEAD"

            # Get commits in the range
            result = subprocess.run([
                "git", "log", commit_range,
                "--pretty=format:%h|%s|%an|%ad",
                "--date=short"
            ], cwd=self.repo_path, capture_output=True, text=True, check=True)

            if not result.stdout.strip():
                return "No changes since last release.\n"

            # Parse commits and categorize
            features = []
            fixes = []
            docs = []
            other = []

            for line in result.stdout.strip().split("\n"):
                parts = line.split("|", 3)
                if len(parts) != 4:
                    continue

                commit_hash, message, author, date = parts

                # Categorize based on commit message
                if message.startswith(("feat:", "feature:")):
                    features.append(f"- {message} ({commit_hash})")
                elif message.startswith(("fix:", "bug:")):
                    fixes.append(f"- {message} ({commit_hash})")
                elif message.startswith(("docs:", "doc:")):
                    docs.append(f"- {message} ({commit_hash})")
                else:
                    other.append(f"- {message} ({commit_hash})")

            # Generate changelog
            changelog_lines = [
                "# Changelog",
                "",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d')}",
                "",
            ]

            if features:
                changelog_lines.extend([
                    "## ✨ New Features",
                    ""
                ])
                changelog_lines.extend(features)
                changelog_lines.append("")

            if fixes:
                changelog_lines.extend([
                    "## 🐛 Bug Fixes",
                    ""
                ])
                changelog_lines.extend(fixes)
                changelog_lines.append("")

            if docs:
                changelog_lines.extend([
                    "## 📚 Documentation",
                    ""
                ])
                changelog_lines.extend(docs)
                changelog_lines.append("")

            if other:
                changelog_lines.extend([
                    "## 🔧 Other Changes",
                    ""
                ])
                changelog_lines.extend(other)
                changelog_lines.append("")

            return "\n".join(changelog_lines)

        except subprocess.CalledProcessError as e:
            print(f"Error generating changelog: {e}")
            return ""

    def check_repository_health(self) -> Dict[str, any]:
        """Check overall repository health.
        
        Returns:
            Dictionary with health check results
        """
        print("Checking repository health...")

        health = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_score": 0,
            "recommendations": []
        }

        # Check if essential files exist
        essential_files = [
            "README.md",
            "LICENSE",
            "pyproject.toml",
            ".gitignore",
            ".github/workflows/ci.yml",
            "tests/",
            "docs/"
        ]

        missing_files = []
        for file_path in essential_files:
            if not (self.repo_path / file_path).exists():
                missing_files.append(file_path)

        health["checks"]["essential_files"] = {
            "score": 100 - (len(missing_files) * 100 / len(essential_files)),
            "missing": missing_files
        }

        if missing_files:
            health["recommendations"].append(
                f"Add missing essential files: {', '.join(missing_files)}"
            )

        # Check git history
        try:
            result = subprocess.run([
                "git", "log", "--oneline", "--since=30 days ago"
            ], check=False, cwd=self.repo_path, capture_output=True, text=True)

            recent_commits = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            health["checks"]["recent_activity"] = {
                "score": min(100, recent_commits * 5),  # 20 commits = 100 score
                "recent_commits": recent_commits
            }

            if recent_commits < 5:
                health["recommendations"].append("Increase development activity")

        except subprocess.CalledProcessError:
            health["checks"]["recent_activity"] = {"score": 0, "error": "Could not check git history"}

        # Check for security files
        security_files = [
            "SECURITY.md",
            ".github/dependabot.yml",
            ".github/workflows/security-scan.yml"
        ]

        present_security_files = sum(
            1 for file_path in security_files
            if (self.repo_path / file_path).exists()
        )

        health["checks"]["security_setup"] = {
            "score": (present_security_files / len(security_files)) * 100,
            "files_present": present_security_files,
            "total_files": len(security_files)
        }

        if present_security_files < len(security_files):
            health["recommendations"].append("Improve security configuration")

        # Calculate overall score
        scores = [check["score"] for check in health["checks"].values() if "score" in check]
        health["overall_score"] = sum(scores) / len(scores) if scores else 0

        return health


class CodeQualityChecker:
    """Handles automated code quality checks and improvements."""

    def __init__(self, repo_path: str = "."):
        """Initialize the code quality checker.
        
        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)

    def run_quality_checks(self) -> Dict[str, any]:
        """Run comprehensive code quality checks.
        
        Returns:
            Dictionary with quality check results
        """
        print("Running code quality checks...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_score": 0,
            "issues": []
        }

        # Check formatting with black
        try:
            result = subprocess.run([
                "black", "--check", "--diff", "."
            ], check=False, cwd=self.repo_path, capture_output=True, text=True)

            results["checks"]["formatting"] = {
                "passed": result.returncode == 0,
                "issues": result.stdout.count("would reformat") if result.stdout else 0
            }

            if result.returncode != 0:
                results["issues"].append("Code formatting issues found (run 'black .' to fix)")

        except FileNotFoundError:
            results["checks"]["formatting"] = {"error": "black not installed"}

        # Check imports with isort
        try:
            result = subprocess.run([
                "isort", "--check-only", "--diff", "."
            ], check=False, cwd=self.repo_path, capture_output=True, text=True)

            results["checks"]["import_sorting"] = {
                "passed": result.returncode == 0,
                "issues": result.stdout.count("Fixing") if result.stdout else 0
            }

            if result.returncode != 0:
                results["issues"].append("Import sorting issues found (run 'isort .' to fix)")

        except FileNotFoundError:
            results["checks"]["import_sorting"] = {"error": "isort not installed"}

        # Check linting with ruff
        try:
            result = subprocess.run([
                "ruff", "check", "."
            ], check=False, cwd=self.repo_path, capture_output=True, text=True)

            issue_count = result.stdout.count("\n") if result.stdout else 0
            results["checks"]["linting"] = {
                "passed": result.returncode == 0,
                "issues": issue_count
            }

            if result.returncode != 0:
                results["issues"].append(f"Linting issues found: {issue_count}")

        except FileNotFoundError:
            results["checks"]["linting"] = {"error": "ruff not installed"}

        # Check type hints with mypy
        try:
            result = subprocess.run([
                "mypy", "."
            ], check=False, cwd=self.repo_path, capture_output=True, text=True)

            error_count = result.stdout.count("error:") if result.stdout else 0
            results["checks"]["type_checking"] = {
                "passed": result.returncode == 0,
                "errors": error_count
            }

            if result.returncode != 0:
                results["issues"].append(f"Type checking errors found: {error_count}")

        except FileNotFoundError:
            results["checks"]["type_checking"] = {"error": "mypy not installed"}

        # Calculate overall score
        passed_checks = sum(
            1 for check in results["checks"].values()
            if isinstance(check, dict) and check.get("passed", False)
        )
        total_checks = len([
            check for check in results["checks"].values()
            if isinstance(check, dict) and "error" not in check
        ])

        results["overall_score"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        return results

    def auto_fix_issues(self) -> bool:
        """Automatically fix code quality issues where possible.
        
        Returns:
            True if fixes were applied successfully
        """
        print("Auto-fixing code quality issues...")

        success = True

        # Fix formatting with black
        try:
            subprocess.run([
                "black", "."
            ], cwd=self.repo_path, check=True)
            print("✅ Code formatting fixed")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Could not fix formatting: {e}")
            success = False

        # Fix import sorting with isort
        try:
            subprocess.run([
                "isort", "."
            ], cwd=self.repo_path, check=True)
            print("✅ Import sorting fixed")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Could not fix import sorting: {e}")
            success = False

        # Fix linting issues with ruff (where possible)
        try:
            subprocess.run([
                "ruff", "check", "--fix", "."
            ], cwd=self.repo_path, check=True)
            print("✅ Auto-fixable linting issues resolved")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  Some linting issues require manual attention: {e}")

        return success


class DocumentationManager:
    """Handles automated documentation tasks."""

    def __init__(self, repo_path: str = "."):
        """Initialize the documentation manager.
        
        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)

    def update_readme_badges(self) -> bool:
        """Update badges in README.md.
        
        Returns:
            True if README was updated successfully
        """
        readme_path = self.repo_path / "README.md"
        if not readme_path.exists():
            print("README.md not found")
            return False

        print("Updating README badges...")

        try:
            with open(readme_path, "r") as f:
                content = f.read()

            # Define badges to add/update
            badges = {
                "python": "![Python](https://img.shields.io/badge/python-3.8%2B-blue)",
                "license": "![License](https://img.shields.io/badge/license-MIT-green)",
                "build": "![Build Status](https://github.com/danieleschmidt/spin-torque-rl-gym/workflows/CI/badge.svg)",
                "coverage": "![Coverage](https://codecov.io/gh/danieleschmidt/spin-torque-rl-gym/branch/main/graph/badge.svg)",
                "docs": "![Documentation](https://readthedocs.org/projects/spin-torque-rl-gym/badge/?version=latest)"
            }

            # Find existing badges section or create one
            badge_section = "\n".join(badges.values()) + "\n"

            # Look for existing badges section
            badge_pattern = r"!\[.*?\]\(.*?\)"
            existing_badges = re.findall(badge_pattern, content)

            if existing_badges:
                # Replace existing badges
                for badge in existing_badges:
                    content = content.replace(badge, "", 1)

                # Add new badges at the top after title
                lines = content.split("\n")
                title_line = -1
                for i, line in enumerate(lines):
                    if line.startswith("# "):
                        title_line = i
                        break

                if title_line >= 0:
                    lines.insert(title_line + 2, badge_section)
                    content = "\n".join(lines)
            else:
                # Add badges after the first heading
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("# "):
                        lines.insert(i + 2, badge_section)
                        break
                content = "\n".join(lines)

            # Write updated content
            with open(readme_path, "w") as f:
                f.write(content)

            print("✅ README badges updated")
            return True

        except Exception as e:
            print(f"❌ Error updating README badges: {e}")
            return False

    def check_broken_links(self) -> List[str]:
        """Check for broken links in documentation.
        
        Returns:
            List of broken links found
        """
        print("Checking for broken links...")

        broken_links = []

        # Find all markdown files
        md_files = list(self.repo_path.rglob("*.md"))

        # Pattern to find markdown links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                links = re.findall(link_pattern, content)

                for link_text, link_url in links:
                    # Skip anchors and mailto links
                    if link_url.startswith(("#", "mailto:")):
                        continue

                    # Check if it's a local file
                    if not link_url.startswith(("http://", "https://")):
                        # Resolve relative path
                        link_path = (md_file.parent / link_url).resolve()
                        if not link_path.exists():
                            broken_links.append(f"{md_file}:{link_url} (file not found)")
                    else:
                        # Check external URL (with timeout to avoid hanging)
                        try:
                            response = requests.head(link_url, timeout=10, allow_redirects=True)
                            if response.status_code >= 400:
                                broken_links.append(f"{md_file}:{link_url} (HTTP {response.status_code})")
                        except requests.RequestException:
                            broken_links.append(f"{md_file}:{link_url} (connection failed)")

            except Exception as e:
                print(f"Warning: Could not check {md_file}: {e}")

        return broken_links


def main():
    """Main entry point for automation helpers."""
    import argparse

    parser = argparse.ArgumentParser(description="Repository automation helpers")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Repository maintenance commands
    repo_parser = subparsers.add_parser("repo", help="Repository maintenance")
    repo_subparsers = repo_parser.add_subparsers(dest="repo_command")

    cleanup_parser = repo_subparsers.add_parser("cleanup", help="Clean up old branches")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Age threshold in days")

    update_parser = repo_subparsers.add_parser("update", help="Update dependencies")
    update_parser.add_argument("--type", choices=["all", "security", "minor", "patch"],
                              default="all", help="Type of updates")

    repo_subparsers.add_parser("health", help="Check repository health")
    repo_subparsers.add_parser("changelog", help="Generate changelog")

    # Code quality commands
    quality_parser = subparsers.add_parser("quality", help="Code quality checks")
    quality_subparsers = quality_parser.add_subparsers(dest="quality_command")

    quality_subparsers.add_parser("check", help="Run quality checks")
    quality_subparsers.add_parser("fix", help="Auto-fix quality issues")

    # Documentation commands
    docs_parser = subparsers.add_parser("docs", help="Documentation management")
    docs_subparsers = docs_parser.add_subparsers(dest="docs_command")

    docs_subparsers.add_parser("badges", help="Update README badges")
    docs_subparsers.add_parser("links", help="Check for broken links")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "repo":
            maintainer = RepositoryMaintainer()

            if args.repo_command == "cleanup":
                deleted = maintainer.cleanup_old_branches(args.days)
                print(f"Deleted {len(deleted)} old branches")

            elif args.repo_command == "update":
                success = maintainer.update_dependencies(args.type)
                print("Dependencies updated" if success else "Update failed")

            elif args.repo_command == "health":
                health = maintainer.check_repository_health()
                print(f"Repository health score: {health['overall_score']:.1f}/100")
                if health["recommendations"]:
                    print("Recommendations:")
                    for rec in health["recommendations"]:
                        print(f"  - {rec}")

            elif args.repo_command == "changelog":
                changelog = maintainer.generate_changelog()
                print(changelog)

        elif args.command == "quality":
            checker = CodeQualityChecker()

            if args.quality_command == "check":
                results = checker.run_quality_checks()
                print(f"Code quality score: {results['overall_score']:.1f}/100")
                if results["issues"]:
                    print("Issues found:")
                    for issue in results["issues"]:
                        print(f"  - {issue}")

            elif args.quality_command == "fix":
                success = checker.auto_fix_issues()
                print("Auto-fix completed" if success else "Some issues require manual attention")

        elif args.command == "docs":
            doc_manager = DocumentationManager()

            if args.docs_command == "badges":
                success = doc_manager.update_readme_badges()
                print("README badges updated" if success else "Failed to update badges")

            elif args.docs_command == "links":
                broken_links = doc_manager.check_broken_links()
                if broken_links:
                    print(f"Found {len(broken_links)} broken links:")
                    for link in broken_links:
                        print(f"  - {link}")
                else:
                    print("No broken links found")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
