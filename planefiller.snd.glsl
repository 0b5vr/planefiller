#version 430

#define lofi(i, m) (floor((i) / (m)) * (m))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define tri(p) (1.-4.*abs(fract(p)-0.5))

const float PI = acos( -1.0 );
const float TAU = PI * 2.0;

const float B2T = 0.43;
const float SAMPLES_PER_SEC = 48000.0;
const float SWING = 0.62;

int SAMPLES_PER_BEAT = int(SAMPLES_PER_SEC * B2T);

// https://www.shadertoy.com/view/XlXcW4
vec3 hash3f(vec3 s) {
  uvec3 r = floatBitsToUint(s);
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  return vec3(r) / float(-1u);
}

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D( float x ) {
  vec2 v = cis(x);
  return mat2(v.x, v.y, -v.y, v.x);
}

layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
  #pragma work_around_begin:layout(std430,binding=0)buffer ssbo{vec2 %s[];};layout(local_size_x=1)in;
  vec2 waveOutSamples[];
  #pragma work_around_end
#else
  layout(std430, binding = 0) buffer SoundOutput{ vec2 waveOutSamples[]; };
  layout(local_size_x = 1) in;
#endif


// == rhythm stuff =================================================================================
float t2sSwing(float t) {
  float st = 4.0 * t / B2T;
  return 2.0 * floor(st / 2.0) + step(SWING, fract(0.5 * st));
}

float s2tSwing(float st) {
  return 0.5 * B2T * (floor(st / 2.0) + SWING * mod(st, 2.0));
}

vec4 seq16(float t, int seq) {
  t = mod(t, 4.0 * B2T);
  int sti = int(t2sSwing(t));
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float i_prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - i_prevStepBehind;
  float prevTime = s2tSwing(prevStep);
  float i_nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + i_nextStepForward;
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
}

// == osc stuff ====================================================================================
vec2 shotgun( float t, float spread ) {
  vec2 sum = vec2(0.0);

  for (int i = 0; i ++ < 64;) {
    vec3 dice = hash3f(i + vec3(7, 1, 3));
    sum += vec2(sin(TAU * t * exp2(spread * dice.x))) * rotate2D(TAU * dice.y);
  }

  return sum / 64.0;
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
}

vec3 cyclic(vec3 p, float pers, float lacu) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  for (int i = 0; i ++ < 5;) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum /= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

// == main =========================================================================================
void main() {
  const float i_TENKAI_HELLO_RIM = 32.0;
  const float i_TENKAI_HELLO_RAVE = 32.0;
  const float i_TENKAI_HELLO_KICK = i_TENKAI_HELLO_RAVE + 32.0;
  const float i_TENKAI_HELLO_BASS = i_TENKAI_HELLO_RAVE + 64.0;
  const float i_TENKAI_HELLO_HIHAT = i_TENKAI_HELLO_BASS;
  const float i_TENKAI_HELLO_FMPERC = i_TENKAI_HELLO_BASS;
  const float i_TENKAI_HELLO_HIHAT_16TH = i_TENKAI_HELLO_BASS + 32.0;
  const float i_TENAKI_HELLO_CLAP = i_TENKAI_HELLO_BASS + 32.0;
  const float i_TENKAI_HELLO_OH = i_TENAKI_HELLO_CLAP + 32.0;
  const float i_TENKAI_BREAK = i_TENKAI_HELLO_OH + 32.0;
  const float i_TENKAI_FULLHOUSE = i_TENKAI_HELLO_OH + 64.0;
  const float i_TENKAI_TRANS = i_TENKAI_FULLHOUSE + 64.0;
  const float i_TENKAI_OUTRO = i_TENKAI_TRANS + 64.0;
  const float i_TENKAI_FADEOUT0 = i_TENKAI_OUTRO + 8.0;
  const float i_TENKAI_FADEOUT1 = i_TENKAI_OUTRO + 32.0;

  int frame = int(gl_GlobalInvocationID.x) + waveOutPosition;
  vec4 time = vec4(frame % (SAMPLES_PER_BEAT * ivec4(1, 4, 32, 65536))) / SAMPLES_PER_SEC;
  float beats = time.w / B2T;
  float beatsbar = lofi(beats, 4.0);
  float beats8bar = lofi(beats, 32.0);

  bool i_condKickHipass = (
    ((i_TENKAI_HELLO_BASS - 3.0) <= beats && beats < i_TENKAI_HELLO_BASS) ||
    (i_TENKAI_BREAK <= beats && beats < i_TENKAI_FULLHOUSE - 4.0)
  );

  int i_patternKick =
    beatsbar == (i_TENKAI_HELLO_OH - 4.0) ? 0x88a6 :
    beatsbar == (i_TENKAI_FULLHOUSE - 4.0) ? 0x808f :
    beatsbar == (i_TENKAI_TRANS - 4.0) ? 0x809e :
    beatsbar == (i_TENKAI_OUTRO - 4.0) ? 0xa18e :
    0x8888;

  float i_timeCrash = float(frame - SAMPLES_PER_BEAT * int(
    i_TENKAI_OUTRO <= beats ? i_TENKAI_OUTRO :
    i_TENKAI_OUTRO - 4.0 <= beats ? i_TENKAI_OUTRO - 4.0 :
    i_TENKAI_TRANS <= beats ? i_TENKAI_TRANS :
    i_TENKAI_TRANS - 4.0 <= beats ? i_TENKAI_TRANS - 4.0 :
    i_TENKAI_FULLHOUSE <= beats ? i_TENKAI_FULLHOUSE :
    i_TENKAI_FULLHOUSE - 4.0 <= beats ? i_TENKAI_FULLHOUSE - 4.0 :
    i_TENKAI_HELLO_OH <= beats ? i_TENKAI_HELLO_OH :
    i_TENKAI_HELLO_BASS <= beats ? i_TENKAI_HELLO_BASS :
    i_TENKAI_HELLO_KICK <= beats ? i_TENKAI_HELLO_KICK :
    -111.0
  )) / SAMPLES_PER_SEC;

  float i_volumeSnare = (
    beats8bar == i_TENKAI_BREAK ? smoothstep(0.0, 32.0 * B2T, time.z) :
    beatsbar == (i_TENKAI_FULLHOUSE - 4.0) ? smoothstep(-4.0 * B2T, 4.0 * B2T, time.y) :
    beatsbar == (i_TENKAI_TRANS - 4.0) ? smoothstep(-4.0 * B2T, 4.0 * B2T, time.y) :
    beatsbar == (i_TENKAI_OUTRO - 4.0) ? smoothstep(-4.0 * B2T, 4.0 * B2T, time.y) :
    0.0
  );

  bool i_rollSnare = (
    beats8bar == i_TENKAI_BREAK ? 28.0 * B2T < time.z :
    2.0 * B2T < time.y
  );

  vec2 dest = vec2(0);
  float sidechain = 1.0;

  if (i_TENKAI_HELLO_KICK <= beats) { // kick
    vec4 seq = seq16(time.y, i_patternKick);
    float t = seq.y;
    float q = seq.w;

    float env = smoothstep(0.3, 0.2, t);

    if (i_condKickHipass) { // hi-pass like
      env *= exp2(-60.0 * t);
    }

    vec2 wave = vec2(0.0);
    vec2 phase = vec2(40.0 * t);
    phase -= 9.0 * exp2(-25 * t);
    phase -= 3.0 * exp2(-50 * t);
    phase -= 3.0 * exp2(-500 * t);
    wave += sin(TAU * phase);

    dest += 0.5 * env * tanh(1.3 * wave);

    sidechain = smoothstep(0.0, 0.7 * B2T, time.x) * smoothstep(0.0, 0.01 * B2T, B2T - time.x);
    sidechain *= 0.5 + 0.5 * smoothstep(0.0, 0.7 * B2T, t) * smoothstep(0.0, 0.01 * B2T, q);
  }

  if (i_TENKAI_HELLO_HIHAT <= beats && beats < i_TENKAI_OUTRO) { // hihat
    int i_patternCH =
      i_TENKAI_HELLO_HIHAT_16TH <= beats ? 0xffff :
      0xeaaa;
    vec4 seq = seq16(time.y, i_patternCH);
    float t = seq.y;

    float vel = fract(seq.x * 0.38);
    float env = exp2(-exp2(6.0 - 1.0 * vel - float(mod(seq.x, 4.0) == 2.0)) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0);
    dest += 0.16 * env * mix(0.2, 1.0, sidechain) * tanh(8.0 * wave);
  }

  if (i_TENKAI_HELLO_OH <= beats && beats < i_TENKAI_OUTRO) { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.y;

    vec2 sum = vec2(0.0);

    for (int i = 0; i ++ < 8;) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.1 * exp2(-14.0 * t) * sidechain * tanh(2.0 * sum);
  }

  if (i_TENKAI_HELLO_FMPERC <= beats && beats < i_TENKAI_OUTRO) { // fm perc
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.y;
    float q = seq.w;
    vec3 dice = hash3f(vec3(seq.x, mod(beatsbar, 32.0), 1.0));

    float freq = exp2(9.0 + 2.0 * dice.x);
    float env = exp2(-exp2(3.0 + 5.0 * dice.y) * t) * smoothstep(0.0, 0.01, q);
    float fm = env * exp2(2.0 + 4.0 * dice.z) * sin(freq * exp2(-t));
    float wave = sin(fm);
    dest += 0.05 * sidechain * vec2(wave) * rotate2D(seq.x);
  }

  if (i_TENKAI_HELLO_RIM <= beats && beats < i_TENKAI_OUTRO) { // rim
    vec4 seq = seq16(time.y, 0x6db7);
    float t = seq.y;

    float env = step(0.0, t) * exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * vec2(wave) * rotate2D(seq.x);
  }

  if (i_TENKAI_FULLHOUSE <= beats && beats < i_TENKAI_OUTRO) { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t = seq.y;
    float q = seq.w;

    float env = exp2(-4.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);

    for (int i = 0; i ++ < 8;) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 2.9 * env * sin(wave + exp2(13.10 + 0.4 * dice.x) * t + dice2.xy);
      wave = 2.8 * env * sin(wave + exp2(14.97 + 0.4 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(14.09 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.05 * env * mix(0.3, 1.0, sidechain) * tanh(sum);
  }

  if (i_TENAKI_HELLO_CLAP <= beats && beats < i_TENKAI_OUTRO) { // clap
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.y;
    float q = seq.w;

    float env = mix(
      exp2(-80.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-100.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

    dest += 0.15 * tanh(20.0 * env * wave);
  }

  if (i_volumeSnare > 0.0) { // snare909
    vec4 seq = seq16(time.y, 0xffff);
    float t = i_rollSnare
      ? mod(time.y, B2T / 6.0)
      : seq.y;

    float env = exp(-20.0 * t);

    vec2 wave = (
      cyclic(vec3(cis(4000.0 * t), 4000.0 * t), 1.0, 2.0).xy
      + sin(1400.0 * t - 40.0 * exp2(-t * 200.0))
    );

    dest += 0.2 * i_volumeSnare * mix(0.3, 1.0, sidechain) * tanh(4.0 * env * wave);
  }

  { // crash
    float t = i_timeCrash;

    float env = mix(exp2(-t), exp2(-14.0 * t), 0.7);
    vec2 wave = shotgun(4000.0 * t, 2.5);
    dest += 0.4 * env * mix(0.1, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // chord stuff
    const int N_CHORD = 8;
    const int CHORD[N_CHORD] = int[](
      0, 7, 10, 12, 15, 17, 19, 22
    );

    float t = mod(time.z, 8.0 * B2T);
    float st = max(1.0, lofi(mod(t2sSwing(t) - 1.0, 32.0), 3.0) + 1.0);
    float stt = s2tSwing(st);
    t = mod(t - stt, 8.0 * B2T);
    float nst = min(st + 3.0, 33.0);
    float nstt = s2tSwing(nst);
    float l = nstt - stt;
    float q = l - t;

    if (beats < i_TENKAI_HELLO_RAVE) {
      t = time.z;
      q = i_TENKAI_HELLO_RAVE * B2T - t;
    }

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    float trans = 3.0 * step(beats, i_TENKAI_TRANS) + step(i_TENKAI_HELLO_RAVE, beats) * step(st, 3.0);

    if (i_TENKAI_HELLO_BASS <= beats) { // bass
      float note = 24.0 + trans + float(CHORD[0]);
      float freq = p2f(note);
      float phase = freq * t;
      float wave = tanh(2.0 * sin(TAU * phase));

      dest += 0.5 * sidechain * env * wave;
    }

    if (beats < i_TENKAI_HELLO_RAVE) { // longnote
      env *= smoothstep(1.0, i_TENKAI_HELLO_RAVE, beats);
    } else { // env
      env *= mix(
        smoothstep(0.6 * l, 0.4 * l, t - 0.4 * l * step(i_TENKAI_FULLHOUSE, beats)),
        exp2(-5.0 * t),
        0.1
      );
    }

    { // choir
      vec2 sum = vec2(0.0);

      for (int i = 0; i ++ < 64;) {
        float fi = float(i);
        vec3 dice = hash3f(i + vec3(8, 4, 2));

        float note = 48.0 + trans + float(CHORD[i % N_CHORD]);
        float freq = p2f(note) * exp2(0.016 * tan(2.0 * dice.y - 1.0));
        float phase = lofi(t * freq, 1.0 / 16.0);

        vec3 c = vec3(0.0);
        vec3 d = vec3(2.0, -3.0, -8.0);
        float k = 0.1 + 0.4 * smoothstep(0.0, i_TENKAI_HELLO_RAVE, beats);
        vec2 wave = cyclic(fract(phase) * d, k, 2.0).xy;

        sum += vec2(wave) * rotate2D(fi);
      }

      dest += 0.05 * mix(0.1, 1.0, sidechain) * env * sum;
    }

    if (i_TENKAI_FULLHOUSE <= beats) { // arp
      int iarp = int(16.0 * t / B2T);
      float note = 48.0 + trans + float(CHORD[iarp % N_CHORD]) + 12.0 * float((iarp % 3) / 2);
      float freq = p2f(note);
      float phase = TAU * lofi(t * freq, 1.0 / 16.0);

      vec2 wave = cyclic(vec3(cis(phase), iarp), 0.5, 2.0).xy * rotate2D(time.w);

      dest += 0.2 * sidechain * env * wave;
    }
  }

  waveOutSamples[frame] = clamp(1.3 * tanh(dest), -1.0, 1.0) * smoothstep(i_TENKAI_FADEOUT1, i_TENKAI_FADEOUT0, beats);
}
