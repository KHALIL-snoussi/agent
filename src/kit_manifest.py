"""
Kit manifest and re-print code system for paint-by-numbers generator.
Handles retrieval codes and kit metadata storage.
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class KitManifest:
    """Kit manifest data structure."""
    retrieval_code: str
    timestamp: str
    input_image: str
    output_pdf: str
    style_preset: Optional[str]
    processing_params: Dict[str, Any]
    color_counts: Dict[str, int]
    canvas_config: Dict[str, Any]
    kit_hash: str


class ManifestManager:
    """Manages kit manifests and retrieval codes."""
    
    def __init__(self, manifest_file: str = "kit_manifests.json"):
        """Initialize manifest manager."""
        self.manifest_file = Path(manifest_file)
        self.manifests = self._load_manifests()
    
    def _load_manifests(self) -> Dict[str, KitManifest]:
        """Load existing manifests from file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {code: KitManifest(**manifest) for code, manifest in data.items()}
            except Exception:
                return {}
        return {}
    
    def _save_manifests(self):
        """Save manifests to file."""
        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            data = {code: asdict(manifest) for code, manifest in self.manifests.items()}
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def generate_retrieval_code(self, input_image: str, style_preset: Optional[str] = None,
                           processing_params: Dict[str, Any] = None,
                           color_counts: Dict[str, int] = None,
                           canvas_config: Dict[str, Any] = None) -> str:
        """
        Generate a unique 8-character retrieval code for a kit.
        
        Args:
            input_image: Path to input image
            style_preset: Name of style preset used
            processing_params: Processing parameters used
            color_counts: Color counts for the kit
            canvas_config: Canvas configuration used
            
        Returns:
            8-character alphanumeric retrieval code
        """
        # Generate base hash from image content and timestamp
        timestamp = datetime.now().isoformat()
        image_hash = self._hash_image_path(input_image, timestamp)
        
        # Create 8-character code
        code = self._create_code_from_hash(image_hash)
        
        # Create manifest
        manifest = KitManifest(
            retrieval_code=code,
            timestamp=timestamp,
            input_image=str(Path(input_image).name),
            output_pdf="",  # Will be set when PDF is generated
            style_preset=style_preset,
            processing_params=processing_params or {},
            color_counts=color_counts or {},
            canvas_config=canvas_config or {},
            kit_hash=image_hash
        )
        
        # Store manifest
        self.manifests[code] = manifest
        self._save_manifests()
        
        return code
    
    def _hash_image_path(self, image_path: str, timestamp: str) -> str:
        """Generate hash from image path and timestamp."""
        combined = f"{image_path}_{timestamp}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _create_code_from_hash(self, hash_string: str) -> str:
        """Create 8-character alphanumeric code from hash."""
        # Use first 8 characters of hash, convert to alphanumeric
        chars = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        
        # Take first 8 bytes of hash
        hash_bytes = hashlib.md5(hash_string.encode()).digest()[:8]
        
        code = ""
        for byte in hash_bytes:
            code += chars[byte % len(chars)]
        
        return code
    
    def get_manifest(self, retrieval_code: str) -> Optional[KitManifest]:
        """Get manifest by retrieval code."""
        return self.manifests.get(retrieval_code)
    
    def update_manifest_pdf_path(self, retrieval_code: str, pdf_path: str):
        """Update the PDF path in an existing manifest."""
        if retrieval_code in self.manifests:
            self.manifests[retrieval_code].output_pdf = str(Path(pdf_path).name)
            self._save_manifests()
    
    def list_manifests(self, limit: int = 10) -> list[KitManifest]:
        """List recent manifests."""
        # Sort by timestamp (newest first)
        sorted_manifests = sorted(
            self.manifests.values(),
            key=lambda m: m.timestamp,
            reverse=True
        )
        return sorted_manifests[:limit]
    
    def validate_retrieval_code(self, retrieval_code: str) -> bool:
        """Validate retrieval code format."""
        if len(retrieval_code) != 8:
            return False
        
        valid_chars = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        return all(char in valid_chars for char in retrieval_code.upper())
    
    def generate_manifest_summary(self) -> str:
        """Generate a summary of all manifests."""
        if not self.manifests:
            return "No kits generated yet."
        
        summary = "📋 Kit Manifest Summary\n" + "="*50 + "\n\n"
        
        for code, manifest in sorted(self.manifests.items(), 
                               key=lambda x: x[1].timestamp, reverse=True):
            # Parse timestamp
            try:
                dt = datetime.fromisoformat(manifest.timestamp)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_time = "Unknown"
            
            summary += f"Code: {code}\n"
            summary += f"  Image: {manifest.input_image}\n"
            summary += f"  Style: {manifest.style_preset or 'Default'}\n"
            summary += f"  Generated: {formatted_time}\n"
            summary += f"  PDF: {manifest.output_pdf or 'Not generated'}\n"
            summary += "-"*30 + "\n"
        
        return summary
    
    def export_legend_data(self, retrieval_code: str, format_type: str = "csv") -> str:
        """Export legend data in CSV or JSON format."""
        manifest = self.get_manifest(retrieval_code)
        if not manifest:
            return "Kit not found."
        
        if format_type.lower() == "csv":
            return self._export_csv(manifest)
        elif format_type.lower() == "json":
            return self._export_json(manifest)
        else:
            return "Unsupported format. Use 'csv' or 'json'."
    
    def _export_csv(self, manifest: KitManifest) -> str:
        """Export manifest data as CSV."""
        csv_lines = ["Color Name,DMC Code,Symbol,Quantity"]
        
        for color_name, count in manifest.color_counts.items():
            # Find DMC code (this would need enhancement to store properly)
            csv_lines.append(f"{color_name},,{count}")
        
        return "\n".join(csv_lines)
    
    def _export_json(self, manifest: KitManifest) -> str:
        """Export manifest data as JSON."""
        return json.dumps(asdict(manifest), indent=2, ensure_ascii=False)
    
    def cleanup_old_manifests(self, days_old: int = 30):
        """Remove manifests older than specified days."""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        to_remove = []
        for code, manifest in self.manifests.items():
            try:
                manifest_time = datetime.fromisoformat(manifest.timestamp).timestamp()
                if manifest_time < cutoff_time:
                    to_remove.append(code)
            except:
                to_remove.append(code)
        
        for code in to_remove:
            del self.manifests[code]
        
        if to_remove:
            self._save_manifests()
            return f"Cleaned up {len(to_remove)} old manifests."
        
        return "No old manifests to clean."
