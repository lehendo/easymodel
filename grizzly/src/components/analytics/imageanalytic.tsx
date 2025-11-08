import React, { useEffect, useRef } from 'react'

export function HeatmapOverlay() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Load placeholder image
    const img = new Image()
    img.src = "https://picsum.photos/400/300"
    img.onload = () => {
      // Draw image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      // Create gradient for heatmap
      const gradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 10,
        canvas.width / 2, canvas.height / 2, canvas.width / 2
      )
      gradient.addColorStop(0, 'rgba(255, 0, 0, 0.8)')
      gradient.addColorStop(0.5, 'rgba(255, 255, 0, 0.5)')
      gradient.addColorStop(1, 'rgba(0, 0, 255, 0)')

      // Apply gradient overlay
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
  }, [])

  return (
    <div className="relative w-full max-w-md mx-auto">
      <canvas ref={canvasRef} width={400} height={300} className="w-full h-auto" />
    </div>
  )
}

