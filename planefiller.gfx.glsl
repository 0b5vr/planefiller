#version 430

layout(binding = 0) uniform sampler2D backBuffer0;
layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
  vec2 resolution = {SCREEN_XRESO, SCREEN_YRESO};
  #define NUM_SAMPLES_PER_SEC 48000.
  float globalTime = waveOutPosition / NUM_SAMPLES_PER_SEC;
#else
  layout(location = 2) uniform float globalTime;
  layout(location = 3) uniform vec2 resolution;
#endif

#if defined(EXPORT_EXECUTABLE)
  #pragma work_around_begin:layout(std430,binding=0)buffer _{vec2 %s[];};
  vec2 waveOutSamples[];
  #pragma work_around_end
#else
  layout(std430, binding = 0) buffer _{ vec2 waveOutSamples[]; };
#endif

out vec4 outColor;

const float PI = acos(-1.0);
const float TAU = PI + PI;
const float FAR = 30.0;

const int SAMPLES_PER_FRAME = 15;
const int TRAVERSAL_STEPS = 30;

#define saturate(x) clamp(x, 0.0, 1.0)
#define linearstep(a, b, x) min(max(((x) - (a)) / ((b) - (a)), 0.0), 1.0)

// == common =======================================================================================
uvec3 seed;

// https://www.shadertoy.com/view/XlXcW4
vec3 hash3f( vec3 s ) {
  uvec3 r = floatBitsToUint( s );
  r = ( ( r >> 16u ) ^ r.yzx ) * 1111111111u;
  r = ( ( r >> 16u ) ^ r.yzx ) * 1111111111u;
  r = ( ( r >> 16u ) ^ r.yzx ) * 1111111111u;
  return vec3( r ) / float( -1u );
}

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D(float t) {
  return mat2(cos(t), -sin(t), sin(t), cos(t));
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
}

// == noise ========================================================================================
vec3 cyclicNoise(vec3 p, float pers) {
  vec4 sum = vec4(0.0);

  for (int i = 0; i ++ < 4;) {
    p *= orthBas(vec3(-1.0, 2.0, -3.0));
    p += sin(p.yzx);
    sum = (sum + vec4(cross(sin(p.zxy), cos(p)), 1.0)) / pers;
    p *= 2.0;
  }

  return sum.xyz / sum.w;
}

// == isects =======================================================================================
vec4 isectBox(vec3 ro, vec3 rd, vec3 s) {
  vec3 xo = -ro / rd;
  vec3 xs = abs(s / rd);

  vec3 dfv = xo - xs;
  vec3 dbv = xo + xs;

  float df = max(max(dfv.x, dfv.y), dfv.z);
  float db = min(min(dbv.x, dbv.y), dbv.z);
  if (df < 0.0 || db < df) { return vec4(FAR); }

  vec3 n = -sign(rd) * step(vec3(df), dfv);
  return vec4(n, df);
}

vec4 isectSphere(vec3 ro, vec3 rd, float r) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - r * r;
  float h = b * b - c;

  float rl = -b - sqrt(h);
  if (h < 0.0 || rl < 0.0) { return vec4(FAR); }

  return vec4(normalize(ro + rd * rl), rl);
}

// == main =========================================================================================
void main() {
  outColor *= 0.0;

  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec2 p = uv - 0.5;
  p.x *= resolution.x / resolution.y;

  vec3 seed = hash3f(vec3(p, globalTime));
  float time = globalTime;
  float beat = time * 135.0 / 60.0;

  for (int i = 0; i ++ < SAMPLES_PER_FRAME;) {
    vec3 colRem = mix(
      vec3(-0.0001, 0.0001, 1.0),
      vec3(-0.4, 0.1, 1.0),
      smoothstep(31.5, 32.5, beat) * smoothstep(224.5, 223.5, beat)
    );

    vec3 ro = orthBas(colRem) * vec3(1.5 * (p + (seed = hash3f(seed)).xy / resolution.y) + vec2(0, 0.1 * time), 6.0) - vec3(0.0, 0.0, 2.0);
    vec3 rd = orthBas(colRem) * vec3(0.0, 0.0, -1.0);

    colRem /= colRem;

    for (int i = 0; i ++ < TRAVERSAL_STEPS;) {
      mat3 material = mat3(
        vec3(0.0),
        vec3(0.0),
        vec3(0.0, 1.0, 0.0)
      );

      vec4 isect = isectBox(ro + vec3(0.0, 0.0, 5.0), rd, vec3(1E3, 1E3, 0.01));
      vec4 isect2;

      {
        // quadtree subdivision
        ro += rd * 0.001;

        const float QUADTREE_SIZE = 0.5;
        const float QUADTREE_DEPTH = 6.0;

        float gen;
        vec3 cell = vec3(0.0, 0.0, 0.5 * FAR);
        vec3 cellSize = vec3(FAR);
        vec3 cellDice = vec3(1e9);

        if (ro.z < 0.0) {
          cellSize = vec3(QUADTREE_SIZE, QUADTREE_SIZE, QUADTREE_DEPTH);
          for (int i = 0; i ++ < 5; ) {
            cellSize.xy *= 0.5 + 0.5 * step(gen < 96.0 ? 0.3 : gen < 160.0 ? 0.4 : gen < 256.0 ? 0.5 : 0.3, cellDice.xy);
            cellSize.z = QUADTREE_DEPTH;

            cell = (floor(ro / cellSize) + 0.5) * cellSize;
            cell.z = 0.0;

            gen = floor(
              dot(cell + cellSize / 2.0 - vec3(0.0, 0.1 * time, 0.0), vec3(-0.03, 0.3, 0.0)) + beat
            );
            cellDice = hash3f(cell + clamp(gen, 31.0, 288.0));
          }
        }

        ro -= rd * 0.001;

        {
          // quadtree traversal
          vec3 i_src = -( ro - cell ) / rd;
          vec3 i_dst = abs( 0.5 * cellSize / rd );
          vec3 bvOrRot = i_src + i_dst;
          float distToNextCell = min(min(bvOrRot.x, bvOrRot.y), bvOrRot.z);

          vec3 rand = vec3(0.0);

          // scene
          bvOrRot = ro - cell + vec3(0.0, 0.0, 5.5 + cellDice.y);

          cellDice = hash3f(cellDice);

          isect2 = isectBox(
            bvOrRot,
            rd,
            vec3(0.5 * cellSize.xy - 0.004, 4.0)
          );

          if (cell.z != 0.0) {
            // skip

          } else if (cellSize.x == cellSize.y && cellSize.x < 1.0 && cellDice.x < 0.1) { // sphere
            if (isect2.w < isect.w) {
              isect = isect2;

              material = mat3(
                vec3(0.04),
                vec3(0.0),
                vec3(0.1, 0.0, 0.0)
              );
            }

            vec3 rotSphere = bvOrRot - vec3(0.0, 0.0, 4.0 + 0.5 * cellSize);
            isect2 = isectSphere(
              rotSphere,
              rd,
              0.4 * min(cellSize.x, cellSize.y)
            );

            if (isect2.w < isect.w) {
              isect = isect2;

              vec3 i_noise = cyclicNoise(4.0 * (ro + rd * isect.w + cellDice), 0.5);
              // vec3 coord = ((rotSphere + rd * isect.w) * orthBas(vec3(cis(time + TAU * cellDice.z), 3.0))) / cellSize.x;
              // material = mat3(
              //   mix(
              //     vec3(0.8, 0.6, 0.02),
              //     vec3(0.02),
              //     step(0.0, coord.z) * (
              //       step(length(abs(coord.xy * vec2(2.0, 1.0) - vec2(0.0, 0.15)) - vec2(0.2, 0.0)), 0.1)
              //       + step(abs(length(coord.xy) - 0.24), step(1.9, abs(atan(coord.x, coord.y))) * 0.01 + step(abs(abs(atan(coord.x, coord.y)) - 1.9), 0.05) * 0.02)
              //     )
              //   ),
              //   vec3(0.0),
              //   vec3(0.1, 0.0, 0.0)
              // );

              material = mat3(
                vec3(0.63, 0.65, 0.66),
                vec3(0.0),
                vec3(0.1, 1.0, 0.0)
              );
            }
          } else if (cellSize.x == cellSize.y && cellSize.x < 1.0 && cellDice.x < (gen > 63.0 ? 0.8 : 0.3)) { // holo
            if (isect2.w < isect.w) {
              isect = isect2;

              material = mat3(
                vec3(0.4),
                vec3(0.0),
                vec3(0.04, 1.0, 0.0)
              );
            }

            vec3 rotPlane = bvOrRot - vec3(0.0, 0.0, 4.02);
            isect2 = isectBox(
              rotPlane,
              rd,
              vec3(cellSize.xy, 0.001)
            );
            if (isect2.w < isect.w) {
              vec3 coord = (bvOrRot + rd * isect2.w);
              vec2 ncoord = ((coord / (0.5 * cellSize - 0.004)).xy);

              float mask = step(max(abs(ncoord.x), abs(ncoord.y)), 0.9) * (
                cellDice.y < 0.2 ? step(length(ncoord), 0.9) * step(0.5, length(ncoord)) :
                cellDice.y < 0.4 ? max(step(abs(ncoord.x + ncoord.y), 0.3), step(abs(ncoord.x - ncoord.y), 0.3)) :
                cellDice.y < 0.5 ? max(step(abs(ncoord.y), 0.2), step(abs(ncoord.x), 0.2)) :
                cellDice.y < 0.6 ? max(step(abs(ncoord.y), 0.22) * step(ncoord.x, 0.3), step(abs(abs(ncoord.y) + ncoord.x - 0.6), 0.3) * step(abs(ncoord.y), 0.8)) :
                cellDice.y < 0.7 ? step(abs(abs(ncoord.y) - 0.4), 0.2) :
                cellDice.y < 0.8 ? step(hash3f(floor(ncoord.xyy / 0.45) + cellDice).x, 0.5) * step(length(fract(ncoord / 0.45) - 0.5), 0.4) :
                cellDice.y < 0.9 ? step(max(abs(ncoord.x), abs(ncoord.y)), 0.9) * step(0.6, max(abs(ncoord.x), abs(ncoord.y))) :
                max(step(max(abs(ncoord.x), abs(ncoord.y)), 0.1), step(min(abs(ncoord.x), abs(ncoord.y)), 0.3) * step(0.3, max(abs(ncoord.x), abs(ncoord.y))) * step(max(abs(ncoord.x), abs(ncoord.y)), 0.5))
              );

              if (mask > 0.0) {
                isect = isect2;

                material = mat3(
                  vec3(0.0),
                  (0.54 - 0.5 * cos(9.0 * PI * max(smoothstep(128.5, 127.5, beat), smoothstep(159.5, 160.5, beat)))) * vec3(1.0, 2.0, 3.0) * (1.0 + 0.5 * sin(120.0 * (coord.y + time))),
                  vec3(0.0, 1.0, 0.0)
                );
              }
            }

          } else if (cellSize.x * cellSize.y < 0.125 && cellDice.x < 0.5 * pow(smoothstep(128.0, 160.0, gen), 2.0) * step(gen, 255.0)) { // grafix
            isect2 = isectBox(
              bvOrRot,
              rd,
              vec3(0.5 * cellSize.xy - 0.004, 3.5 + smoothstep(159.5, 160.5, beat))
            );

            if (isect2.w < isect.w) {
              isect = isect2;

              vec3 coord = (bvOrRot + rd * isect.w);
              vec3 i_gridcoord = (coord - vec3(0.0, 0.0, 4.5)) / (cellSize - 0.008) * cellSize;
              vec2 uv = 0.5 + ((coord / (cellSize - 0.008)).xy);
              vec3 grafix =
                cellDice.y < 0.05 ? vec3(1.0) :
                cellDice.y < 0.2 ? max(sin(80.0 * cyclicNoise(5.0 * coord + 8.0 * cellDice, 0.5).x + 20.0 * time + vec3(0.0, 1.0, 2.0)), 0.0) :
                cellDice.y < 0.4 ? saturate(abs(mod(6.0 * (uv.y + time + cellDice.z) + vec3(0, 4, 2), 6.0) - 3.0) - 1.0) :
                cellDice.y < 0.6 ? vec3(dot(1.0 - abs(isect.xyz), 1.0 - step(0.05, abs(fract(64.0 * i_gridcoord + 0.5) - 0.5)))) :
                cellDice.y < 0.8 ? step(0.5, vec3(fract(20.0 * (abs(coord.x) + abs(coord.y) + abs(coord.z) + cellDice.z) - 3.0 * time))) :
                cellDice.y < 0.95 ? vec3(step(hash3f(floor(coord * 256.0) + floor(beat * 4.0) + cellDice).x, 0.4)) :
                vec3(0.02, 0.02, 1.0);

              material = mat3(
                vec3(0.0),
                grafix,
                vec3(0.04, 0.0, 0.0)
              );
            }

          } else {
            if (isect2.w < isect.w) {
              isect = isect2;

              material = mat3(
                vec3(0.8, 0.82, 0.85),
                vec3(0.0),
                vec3(0.2, step(0.5, cellDice.y), 0.0)
              );
            }
          }

          rand += 1.0 + step(0.5, cellDice);

          // should we skip the cell?
          if (distToNextCell < isect.w) {
            ro += distToNextCell * rd;
            continue;
          }
        }
      }

      if (isect.w > FAR - 1.0) {
        break;
      }

      vec3 i_baseColor = material[0];
      vec3 i_emissive = material[1];
      float i_roughness = material[2].x;
      float i_metallic = material[2].y;

      outColor.xyz += colRem * i_emissive;

      // if hit then
      ro += isect.w * rd + isect.xyz * 0.001;
      float sqRoughness = i_roughness * i_roughness;
      float sqSqRoughness = sqRoughness * sqRoughness;
      float halfSqRoughness = 0.5 * sqRoughness;

      {
        float NdotV = dot( isect.xyz, -rd );
        float Fn = mix( 0.04, 1.0, pow( 1.0 - NdotV, 5.0 ) );
        float spec = max(
          step((seed = hash3f(seed)).x, Fn), // non metallic, fresnel
          i_metallic // metallic
        );

        // sample ggx or lambert
        seed.y = sqrt( ( 1.0 - seed.y ) / ( 1.0 - spec * ( 1.0 - sqSqRoughness ) * seed.y ) );
        vec3 woOrH = orthBas( isect.xyz ) * vec3(
          sqrt( 1.0 - seed.y * seed.y ) * sin( TAU * seed.z + vec2( 0.0, TAU / 4.0 ) ),
          seed.y
        );

        if (spec > 0.0) {
          // specular
          // note: woOrH is H right now
          vec3 i_H = woOrH;
          vec3 i_wo = reflect(rd, i_H);
          if (dot(i_wo, isect.xyz) < 0.0) {
            break;
          }

          // vector math
          float NdotL = dot( isect.xyz, i_wo );
          float i_VdotH = dot( -rd, i_H );
          float i_NdotH = dot( isect.xyz, i_H );

          // fresnel
          vec3 i_F0 = mix(vec3(0.04), i_baseColor, i_metallic);
          vec3 i_Fh = mix(i_F0, vec3(1.0), pow(1.0 - i_VdotH, 5.0));

          // brdf
          // colRem *= Fh / Fn * G * VdotH / ( NdotH * NdotV );
          colRem *= max(
            i_Fh / mix(Fn, 1.0, i_metallic)
              / ( NdotV * ( 1.0 - halfSqRoughness ) + halfSqRoughness ) // G1V / NdotV
              * NdotL / ( NdotL * ( 1.0 - halfSqRoughness ) + halfSqRoughness ) // G1L
              * i_VdotH / i_NdotH,
            0.0
          );

          // wo is finally wo
          woOrH = i_wo;
        } else {
          // diffuse
          // note: woOrH is wo right now
          if (dot(woOrH, isect.xyz) < 0.0) {
            break;
          }

          // calc H
          // vector math
          vec3 i_H = normalize( -rd + woOrH );
          float i_VdotH = dot( -rd, i_H );

          // fresnel
          float i_Fh = mix(0.04, 1.0, pow(1.0 - i_VdotH, 5.0));

          // brdf
          colRem *= (1.0 - i_Fh) / (1.0 - Fn) * i_baseColor;
        }

        // prepare the rd for the next ray
        rd = woOrH;

        // if the ray goes beind the surface, invalidate it
        colRem *= max(step(0.0, dot(woOrH, isect.xyz)), 0.0);
      }

      if (dot(colRem, colRem) < 0.01) {
        break;
      }
    }

    outColor.xyz += vec3(1.0, 2.0, 3.0) * step(0.0, saturate(rd.z)) * colRem * step(mod(beat - 1.25 + 0.5 * rd.y, 4.0), 0.1) * smoothstep(16.0, 32.0, beat);
  }

  outColor = mix(
    max(sqrt(outColor / float(SAMPLES_PER_FRAME)), 0.0),
    max(texture(backBuffer0, uv), 0.0),
    0.5
  ) * smoothstep(0.0, 16.0, beat) * smoothstep(320.0, 288.0, beat);
}

