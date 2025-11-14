import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { AlertCircle, Zap, Shield, TrendingUp } from 'lucide-react';

const IFGSMVisualizer = () => {
  const [epsilon, setEpsilon] = useState(0.04);
  const [iterations, setIterations] = useState(10);
  const [selectedDigit, setSelectedDigit] = useState(0);
  const [isAttacking, setIsAttacking] = useState(false);
  const [attackResults, setAttackResults] = useState(null);

  const simulateAttack = () => {
    setIsAttacking(true);
    
    setTimeout(() => {
      const originalConfidence = 0.95 + Math.random() * 0.04;
      const adversarialConfidence = Math.random() * 0.3;
      const wrongPrediction = (selectedDigit + Math.floor(Math.random() * 9) + 1) % 10;
      
      setAttackResults({
        original: {
          prediction: selectedDigit,
          confidence: originalConfidence
        },
        adversarial: {
          prediction: wrongPrediction,
          confidence: 0.92 - adversarialConfidence,
          wrongLabel: wrongPrediction
        },
        perturbation: {
          lInf: epsilon,
          l2: epsilon * 0.7 * (1 + Math.random() * 0.3)
        },
        success: true
      });
      setIsAttacking(false);
    }, 1500);
  };

  const getChineseDigit = (digit) => {
    const chars = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    return chars[digit];
  };

  const OriginalImage = ({ digit }) => (
    <div className="relative w-48 h-48 bg-gray-100 rounded-lg border-2 border-gray-300 flex items-center justify-center">
      <div className="text-8xl font-bold text-gray-800">
        {getChineseDigit(digit)}
      </div>
      <div className="absolute bottom-2 right-2 bg-green-500 text-white px-2 py-1 rounded text-xs font-semibold">
        Original
      </div>
    </div>
  );

  const AdversarialImage = ({ digit, attacked }) => (
    <div className="relative w-48 h-48 bg-gray-100 rounded-lg border-2 border-red-400 flex items-center justify-center">
      <div className="text-8xl font-bold text-gray-800 opacity-90">
        {getChineseDigit(digit)}
      </div>
      {attacked && (
        <>
          <div className="absolute inset-0 bg-red-100 opacity-20 rounded-lg" />
          <svg className="absolute inset-0 w-full h-full opacity-30">
            {[...Array(20)].map((_, i) => (
              <line
                key={i}
                x1={Math.random() * 192}
                y1={Math.random() * 192}
                x2={Math.random() * 192}
                y2={Math.random() * 192}
                stroke="red"
                strokeWidth="0.5"
              />
            ))}
          </svg>
        </>
      )}
      <div className="absolute bottom-2 right-2 bg-red-500 text-white px-2 py-1 rounded text-xs font-semibold">
        Adversarial
      </div>
    </div>
  );

  const PerturbationViz = () => (
    <div className="w-48 h-48 bg-gradient-to-br from-blue-200 via-purple-200 to-pink-200 rounded-lg border-2 border-purple-400 flex items-center justify-center relative overflow-hidden">
      <div className="absolute inset-0">
        {[...Array(50)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-white opacity-40"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              width: `${2 + Math.random() * 4}px`,
              height: `${2 + Math.random() * 4}px`,
            }}
          />
        ))}
      </div>
      <div className="text-4xl font-bold text-purple-800 z-10">
        δ
      </div>
      <div className="absolute bottom-2 right-2 bg-purple-500 text-white px-2 py-1 rounded text-xs font-semibold">
        Perturbation
      </div>
    </div>
  );

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="text-yellow-500" />
            IFGSM Attack on Chinese MNIST Classifier
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Attack Parameters */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <label className="text-sm font-semibold">Epsilon (ε)</label>
                    <Slider
                      value={[epsilon]}
                      onValueChange={(val) => setEpsilon(val[0])}
                      min={0.01}
                      max={0.1}
                      step={0.01}
                      className="w-full"
                    />
                    <div className="text-xs text-gray-600 text-center">{epsilon.toFixed(2)}</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <label className="text-sm font-semibold">Iterations</label>
                    <Slider
                      value={[iterations]}
                      onValueChange={(val) => setIterations(val[0])}
                      min={1}
                      max={20}
                      step={1}
                      className="w-full"
                    />
                    <div className="text-xs text-gray-600 text-center">{iterations}</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <label className="text-sm font-semibold">Target Digit</label>
                    <select
                      value={selectedDigit}
                      onChange={(e) => setSelectedDigit(Number(e.target.value))}
                      className="w-full p-2 border rounded"
                    >
                      {[...Array(10)].map((_, i) => (
                        <option key={i} value={i}>
                          {i} ({getChineseDigit(i)})
                        </option>
                      ))}
                    </select>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Attack Button */}
            <div className="flex justify-center">
              <Button
                onClick={simulateAttack}
                disabled={isAttacking}
                className="w-48 h-12 text-lg"
              >
                {isAttacking ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
                    Attacking...
                  </>
                ) : (
                  <>
                    <Zap className="mr-2" />
                    Launch Attack
                  </>
                )}
              </Button>
            </div>

            {/* Visualization */}
            {attackResults && (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex justify-center items-center gap-8">
                  <div className="text-center space-y-2">
                    <OriginalImage digit={selectedDigit} />
                    <div className="text-sm font-semibold">
                      Prediction: {attackResults.original.prediction}
                    </div>
                    <div className="text-xs text-green-600">
                      Confidence: {(attackResults.original.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="text-center space-y-2">
                    <div className="text-4xl text-gray-400">+</div>
                  </div>

                  <div className="text-center space-y-2">
                    <PerturbationViz />
                    <div className="text-sm font-semibold text-purple-600">
                      L∞: {attackResults.perturbation.lInf.toFixed(4)}
                    </div>
                    <div className="text-xs text-purple-600">
                      L2: {attackResults.perturbation.l2.toFixed(4)}
                    </div>
                  </div>

                  <div className="text-center space-y-2">
                    <div className="text-4xl text-gray-400">=</div>
                  </div>

                  <div className="text-center space-y-2">
                    <AdversarialImage 
                      digit={attackResults.adversarial.wrongLabel} 
                      attacked={true} 
                    />
                    <div className="text-sm font-semibold text-red-600">
                      Prediction: {attackResults.adversarial.wrongLabel} ✗
                    </div>
                    <div className="text-xs text-red-600">
                      Confidence: {(attackResults.adversarial.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Statistics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="bg-green-50 border-green-200">
                    <CardContent className="pt-6">
                      <div className="flex items-center gap-2">
                        <Shield className="text-green-600" />
                        <div>
                          <div className="text-sm text-gray-600">Original Accuracy</div>
                          <div className="text-2xl font-bold text-green-600">
                            {(attackResults.original.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-red-50 border-red-200">
                    <CardContent className="pt-6">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="text-red-600" />
                        <div>
                          <div className="text-sm text-gray-600">Attack Success Rate</div>
                          <div className="text-2xl font-bold text-red-600">
                            {attackResults.success ? '100%' : '0%'}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-blue-50 border-blue-200">
                    <CardContent className="pt-6">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="text-blue-600" />
                        <div>
                          <div className="text-sm text-gray-600">Perturbation Magnitude</div>
                          <div className="text-2xl font-bold text-blue-600">
                            ε = {epsilon.toFixed(3)}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Info Box */}
                <Card className="bg-yellow-50 border-yellow-200">
                  <CardContent className="pt-4">
                    <div className="flex gap-2">
                      <AlertCircle className="text-yellow-600 flex-shrink-0 mt-1" size={20} />
                      <div className="text-sm text-gray-700">
                        <strong>IFGSM Attack:</strong> This iterative attack adds small perturbations 
                        over {iterations} iterations with step size ε/{iterations} = {(epsilon/iterations).toFixed(4)}. 
                        The perturbations are imperceptible to humans but cause the model to misclassify 
                        the Chinese character with high confidence.
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Algorithm Explanation */}
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 text-sm">
            <div>
              <h3 className="font-semibold mb-2">IFGSM (Iterative Fast Gradient Sign Method)</h3>
              <p className="text-gray-600">
                An extension of FGSM that applies smaller perturbations iteratively:
              </p>
            </div>
            <div className="bg-gray-100 p-4 rounded font-mono text-xs">
              x₀ = x (original image)<br/>
              for i = 1 to N:<br/>
              &nbsp;&nbsp;xᵢ = xᵢ₋₁ + α · sign(∇ₓL(θ, x, y))<br/>
              &nbsp;&nbsp;xᵢ = clip(xᵢ, x - ε, x + ε)<br/>
              return x_N (adversarial example)
            </div>
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <strong>Parameters:</strong>
                <ul className="list-disc list-inside text-gray-600 mt-1">
                  <li>ε: Maximum perturbation bound</li>
                  <li>α = ε/N: Step size per iteration</li>
                  <li>N: Number of iterations</li>
                </ul>
              </div>
              <div>
                <strong>Key Features:</strong>
                <ul className="list-disc list-inside text-gray-600 mt-1">
                  <li>More effective than single-step FGSM</li>
                  <li>Maintains bounded perturbations</li>
                  <li>Creates stronger adversarial examples</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default IFGSMVisualizer;